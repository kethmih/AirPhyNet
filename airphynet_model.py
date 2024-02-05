import time

import torch
import torch.nn as nn

from torch.nn.modules.rnn import GRU
from ode_func import ODEFunc
from diffeq_solver import DiffeqSolver

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EncoderAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.num_nodes = adj_mx.shape[0]
        self.num_edges = (adj_mx > 0.).sum()
        self.gcn_step = int(model_kwargs.get('gcn_step', 2))
        self.filter_type = model_kwargs.get('filter_type')
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.latent_dim = int(model_kwargs.get('latent_dim', 4))

class AirPhyNetModel(nn.Module, EncoderAttrs):
    def __init__(self, adj_mx, edge_index, edge_attr, logger, **model_kwargs):
        nn.Module.__init__(self)
        EncoderAttrs.__init__(self, adj_mx, **model_kwargs)
        self._logger = logger

        ####################################################
        # RNN Encoder
        ####################################################
        self.encoder_z0 = Encoder_z0_RNN(adj_mx, **model_kwargs)

        ####################################################
        # ODE solver
        ####################################################
        self.n_traj_samples = int(model_kwargs.get('n_traj_samples', 1))
        self.ode_method = model_kwargs.get('ode_method', 'dopri5')
        self.atol = float(model_kwargs.get('odeint_atol', 1e-4))
        self.rtol = float(model_kwargs.get('odeint_rtol', 1e-3))
        self.num_gen_layer = int(model_kwargs.get('gen_layers', 1))
        self.ode_gen_dim = int(model_kwargs.get('gen_dim', 64))
        ode_set_str = "ODE setting --latent {} --samples {} --method {} \
            --atol {:6f} --rtol {:6f} --gen_layer {} --gen_dim {}".format(\
                self.latent_dim, self.n_traj_samples, self.ode_method, \
                self.atol, self.rtol, self.num_gen_layer, self.ode_gen_dim)
        self.adj_mx =  adj_mx
        self.edge_index =  edge_index
        self.edge_attr =  edge_attr

        self._logger.info(ode_set_str)

        self.save_latent = bool(model_kwargs.get('save_latent', False))
        self.latent_feat = None # used to extract the latent feature

        ####################################################
        # Decoder
        ####################################################
        self.horizon = int(model_kwargs.get('horizon', 1))
        self.out_feat = int(model_kwargs.get('output_dim', 1))
        self.decoder = Decoder(
            self.out_feat,
            adj_mx,
            self.num_nodes,
            self.num_edges,
        ).to(device)

    ##########################################
    def forward(self, inputs, labels=None,batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_nodes * input_dim)
        :param labels: shape (horizon, batch_size, num_nodes * output_dim)
        :param batches_seen: batches seen till now
        :return: outputs: (self.horizon, batch_size, self.num_edges * self.output_dim)
        """
        perf_time = time.time()
        # shape: [1, batch, num_nodes * latent_dim]
        first_point_mu, first_point_std, last_wind_vars = self.encoder_z0(inputs)
        self._logger.debug("Recognition complete with {:.1f}s".format(time.time() - perf_time))

        # sample 'n_traj_samples' trajectory
        perf_time = time.time()
        means_z0 = first_point_mu.repeat(self.n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(self.n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)

        time_steps_to_predict = torch.arange(start=0, end=self.horizon, step=1).float().to(device)
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)

        # Shape of sol_ys (horizon, n_traj_samples, batch_size, self.num_nodes * self.latent_dim)
        odefunc = ODEFunc(last_wind_vars, self.ode_gen_dim, self.latent_dim, self.adj_mx, self.edge_index, self.edge_attr,
                          self.gcn_step, self.num_nodes,filter_type=self.filter_type).to(device)
        self.diffeq_solver = DiffeqSolver(odefunc,self.ode_method, self.latent_dim, odeint_rtol=self.rtol,
                                          odeint_atol=self.atol)
        sol_ys, fe = self.diffeq_solver(first_point_enc, time_steps_to_predict)
        self._logger.debug("ODE solver complete with {:.1f}s".format(time.time() - perf_time))
        if(self.save_latent):
            # Shape of latent_feat (horizon, batch_size, self.num_nodes * self.latent_dim)
            self.latent_feat = torch.mean(sol_ys.detach(), axis=1)

        perf_time = time.time()
        outputs = self.decoder(sol_ys)
        self._logger.debug("Decoder complete with {:.1f}s".format(time.time() - perf_time))

        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs, fe

class Encoder_z0_RNN(nn.Module, EncoderAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        EncoderAttrs.__init__(self, adj_mx, **model_kwargs)
        self.recg_type = model_kwargs.get('recg_type', 'gru') # gru

        if(self.recg_type == 'gru'):
            # gru settings
            self.input_var = int(model_kwargs.get('input_var', 3))
            self.input_dim = int(model_kwargs.get('input_dim', 1))
            self.gru_rnn = GRU(self.input_dim, self.rnn_units).to(device)
        else:
            raise NotImplementedError("The recognition net only support 'gru'.")

        # hidden to z0 settings
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.rnn_units, 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim * 2),)

        init_network_weights(self.hiddens_to_z0)

    def forward(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_edges * input_var)
        :return: mean, std: # shape (n_samples=1, batch_size, self.latent_dim)
        """
        if(self.recg_type == 'gru'):
            # shape of outputs: (seq_len, batch, num_senor * rnn_units)
            seq_len, batch_size = inputs.size(0), inputs.size(1)
            inputs = inputs.reshape(seq_len, batch_size, self.num_nodes, self.input_var)
            inputs = inputs.reshape(seq_len, batch_size * self.num_nodes, self.input_var) #(24, 1120, 6)

            pm25 = inputs[:,:,0].unsqueeze(-1)
            wind_vars = inputs[:,:,-2:] 
            outputs, _ = self.gru_rnn(pm25) \

            last_output = outputs[-1]
            last_output = torch.reshape(last_output, (batch_size, self.num_nodes, -1))  # (batch_size, num_nodes, rnn_units) 
            last_wind_vars = torch.reshape(wind_vars[-1], (batch_size, self.num_nodes, -1)) #(batch_size, num_nodes, wind_dim) 
        else:
            raise NotImplementedError("The recognition net only support 'gru'.")

        mean, std = split_last_dim(self.hiddens_to_z0(last_output))
        mean = mean.reshape(batch_size, -1) # (batch_size, num_nodes * latent_dim)(32, 140)
        std = std.reshape(batch_size, -1) # (batch_size, num_nodes * latent_dim)(32, 140)
        std = std.abs()

        assert(not torch.isnan(mean).any())
        assert(not torch.isnan(std).any())

        return mean.unsqueeze(0), std.unsqueeze(0), last_wind_vars # for n_sample traj

class Decoder(nn.Module):
    def __init__(self, output_dim, adj_mx, num_nodes, num_edges):
        super(Decoder, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.output_dim = output_dim

    def forward(self, inputs):
        """
        :param inputs: (horizon, n_traj_samples, batch_size, num_nodes * latent_dim)
        :return outputs: (horizon, batch_size, num_nodes * output_dim), average result of n_traj_samples.
        """
        assert(len(inputs.size()) == 4)
        horizon, n_traj_samples, batch_size = inputs.size()[:3]

        inputs = inputs.reshape(horizon, n_traj_samples, batch_size, self.num_nodes, -1).transpose(-2, -1)
        latent_dim = inputs.size(-2)
        outputs = inputs.reshape(horizon, n_traj_samples, batch_size, latent_dim, self.num_nodes, self.output_dim)

        outputs = torch.mean(
            torch.mean(outputs, axis=3),
            axis=1
        )
        outputs = outputs.reshape(horizon, batch_size, -1)
        return outputs