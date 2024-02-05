import os
import time
from random import SystemRandom
import arrow

import numpy as np
import pandas as pd
import torch

from utils import *
from airphynet_model import AirPhyNetModel
from metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AirPhyNetSupervisor:
    def __init__(self, adj_mx, edge_index, edge_attr, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        #Logging
        self._log_dir = self._get_log_dir(kwargs)
        self.model_dir = kwargs.get('base_model_dir')
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']
        self._logger.info('Scaler mean: {:.6f}, std {:.6f}.'.format(self.standard_scaler.mean, self.standard_scaler.std))

        self.num_nodes = adj_mx.shape[0]
        self.input_var = int(self._model_kwargs.get('input_var', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len', 1))
        self.horizon = int(self._model_kwargs.get('horizon', 1))
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))

        # setup model
        pgdenet_model = AirPhyNetModel(adj_mx, edge_index, edge_attr, self._logger, **self._model_kwargs)
        self.pgdenet_model = pgdenet_model.cuda() if torch.cuda.is_available() else pgdenet_model
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self._logger.info('Loading model...')
            self.load_model()
            
    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            filter_type = kwargs['model'].get('filter_type')
            learning_rate = kwargs['train'].get('base_lr')
            horizon = kwargs['model'].get('horizon')
            
            if filter_type == 'diff':
                filter_type_abbr = 'DiffOnly'
            elif filter_type == 'adv':
                filter_type_abbr = 'AdvOnly'
            elif filter_type == 'diff_adv':
                filter_type_abbr = 'DiffAdv'
            elif filter_type == 'unkP':
                filter_type_abbr = 'NoPhy'
            else:
                filter_type_abbr = 'Default'
            
            run_id = 'airphynet_%s_h_%d_lr_%g_bs_%d_%s/' % (filter_type_abbr, horizon, learning_rate, batch_size,time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch, model_dir):
        model_path = model_dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        config = dict(self._kwargs)
        config['model_state_dict'] = self.pgdenet_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, os.path.join(model_path,'epo%d.tar') % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return os.path.join(model_path,'epo%d.tar') % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.pgdenet_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.pgdenet_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.pgdenet_model(x)
                break

    def train(self, **kwargs):
        self._logger.info('Model mode: train')
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def _train(self, base_lr, steps, patience = 20, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):

        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.pgdenet_model.parameters(), lr = base_lr, eps = epsilon)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

        self._logger.info('Start training ...')
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches: {}".format(num_batches))
        batches_seen = num_batches * self._epoch_num

        # used for nfe
        c = []
        res, keys = [], []

        for epoch_num in range(self._epoch_num, epochs):
            print('\nTrain epoch %s:' % (epoch_num))
            self.pgdenet_model.train()

            train_iterator = self._data['train_loader'].get_iterator()
            losses = []

            start_time = time.time()

            c.clear() #nfe
            for i, (x, y) in enumerate(train_iterator):
                if(i >= num_batches):
                    break
                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)
                output, fe = self.pgdenet_model(x, y, batches_seen)

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters
                    optimizer = torch.optim.Adam(self.pgdenet_model.parameters(), lr=base_lr, eps=epsilon)

                loss = self._compute_loss(y, output)
                self._logger.debug("FE: number - {}, time - {:.3f} s, err - {:.3f}".format(*fe, loss.item()))
                c.append([*fe, loss.item()])

                self._logger.debug(loss.item())
                losses.append(loss.item())

                batches_seen += 1 
                loss.backward()
                optimizer.step()
                
                del x, y, output, loss 
                torch.cuda.empty_cache() 

            # used for nfe
            res.append(pd.DataFrame(c, columns=['nfe', 'time', 'err']))
            keys.append(epoch_num)

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")

            val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)

            end_time = time.time()

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num, self.model_dir)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss


            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

        if bool(self._model_kwargs.get('nfe', False)):
            res = pd.concat(res, keys=keys)
            # self._logger.info("res.shape: ", res.shape)
            res.index.names = ['epoch', 'iter']
            filter_type = self._model_kwargs.get('filter_type', 'unknown')
            atol = float(self._model_kwargs.get('odeint_atol', 1e-5))
            rtol = float(self._model_kwargs.get('odeint_rtol', 1e-5))
            nfe_file = os.path.join(
                self._data_kwargs.get('dataset_dir', 'data'),
                'nfe_{}_a{}_r{}.pkl'.format(filter_type, int(atol*1e5), int(rtol*1e5)))
            res.to_pickle(nfe_file)
            # res.to_csv(nfe_file)

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_nodes, input_dim)
        :param y: shape (batch_size, horizon, num_nodes, input_dim)
        :returns x shape (seq_len, batch_size, num_nodes, input_dim)
                 y shape (horizon, batch_size, num_nodes, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_nodes, input_var)
        :param y: shape (horizon, batch_size, num_nodes, input_var)
        :return: x: shape (seq_len, batch_size, num_nodes * input_var)
                 y: shape (horizon, batch_size, num_nodes * output_dim)
        """
        batch_size = x.size(1)
        self._logger.debug("size of x {}".format(x.size()))
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_var)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)

    def _compute_loss_eval(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true).item(), masked_mape_loss(y_predicted, y_true).item(), masked_rmse_loss(y_predicted, y_true).item()

    def evaluate(self, dataset='val', batches_seen=0, save=False):
        """
        Computes mae rmse mape loss and the predict if save
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.pgdenet_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            mae_losses = []
            mape_losses = []
            rmse_losses = []
            y_dict = None

            if(save):
                y_truths = []
                y_preds = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output, fe = self.pgdenet_model(x)
                mae, mape, rmse = self._compute_loss_eval(y, output)
                mae_losses.append(mae)
                mape_losses.append(mape)
                rmse_losses.append(rmse)

                if(save):
                    y_truths.append(y.cpu())
                    y_preds.append(output.cpu())

            mean_loss = {
                'mae': np.mean(mae_losses),
                'mape': np.mean(mape_losses),
                'rmse': np.mean(rmse_losses)
            }


            self._logger.info('Evaluation: - mae - {:.4f} - mape - {:.4f} - rmse - {:.4f}'.format(mean_loss['mae'], mean_loss['mape'], mean_loss['rmse']))
#             self._writer.add_scalar('{} loss'.format(dataset), mean_loss['mae'], batches_seen)

            if(save):
                y_preds = np.concatenate(y_preds, axis=1)
                y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

                y_truths_scaled = []
                y_preds_scaled = []
                # self._logger.debug("y_preds shape: {}, y_truth shape {}".format(y_preds.shape, y_truths.shape))
                for t in range(y_preds.shape[0]):
                    y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                    y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                    y_truths_scaled.append(y_truth)
                    y_preds_scaled.append(y_pred)

                y_preds_scaled = np.stack(y_preds_scaled)
                y_truths_scaled = np.stack(y_truths_scaled)


                y_dict = {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

                # save_dir = self._data_kwargs.get('dataset_dir', 'data')
                # save_path = os.path.join(save_dir, 'pred.npz')
                # np.savez(save_path, prediction=y_preds_scaled, turth=y_truths_scaled)

            return mean_loss['mae'], y_dict

    def evaluate_more(self, dataset='test', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.pgdenet_model= self.pgdenet_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []

            y_truths = []
            y_preds = []

            mae_losses = []
            mape_losses = []
            rmse_losses = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output, fe = self.pgdenet_model(x)
                
                seq_len=[2, 4, 8, 16, 24]
                mae, mape, rmse = [], [], []

                for seq in seq_len:
                    _mae, _mape, _rmse = self._compute_loss_eval(y[seq-1], output[seq-1]) #y[seq-1] : (32,35)
                    mae.append(_mae)
                    mape.append(_mape)
                    rmse.append(_rmse)

                mae_losses.append(mae)
                mape_losses.append(mape)
                rmse_losses.append(rmse)

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = {
                'mae': np.mean(mae_losses, axis=0),
                'mape': np.mean(mape_losses, axis=0),
                'rmse': np.mean(rmse_losses, axis=0)
            }

            for i, seq in enumerate(seq_len):
                self._logger.info('Evaluation seq {}: - mae - {:.4f} - mape - {:.4f} - rmse - {:.4f}'.format(
                    seq, mean_loss['mae'][i], mean_loss['mape'][i], mean_loss['rmse'][i]))

#             self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            return mean_loss['mae'], mean_loss['mape'], mean_loss['rmse'], y_preds_scaled, y_truths_scaled #{'prediction': y_preds_scaled, 'truth': y_truths_scaled}