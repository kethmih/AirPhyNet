from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from utils import *
from pgdenet_supervisor import PGDENetSupervisor

import numpy as np
import torch

def get_mean_std(data_list):
    return data_list.mean(), data_list.std()

def main(args):
    print("Generating Adjacency Matrix")
    sensor_ids, sensor_id_to_ind, adj_mx, edge_index, edge_attr = get_adjacency_matrix(args)
    
    rmse_list, mae_list, mape_list, preds_list, truths_list = [], [], [], [], []
    for exp_idx in range(args.exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)
        with open(args.config_filename) as f:
            supervisor_config = yaml.load(f,Loader=yaml.Loader)
            
            base_dir = supervisor_config['base_dir']
            supervisor_config['base_dir'] = os.path.join(base_dir, 'exp_'+str(exp_idx)) 
            base_model = supervisor_config['base_model_dir']
            supervisor_config['base_model_dir'] = os.path.join(base_model, 'exp_'+str(exp_idx)) 

            supervisor = PGDENetSupervisor(adj_mx=adj_mx, edge_index=edge_index, edge_attr=edge_attr, **supervisor_config)
            supervisor.train()
            mae, mape, rmse, preds, truths = supervisor.evaluate_more('test')

            mae_list.append(mae)
            mape_list.append(mape)
            rmse_list.append(rmse)
            preds_list.append(preds)
            truths_list.append(truths)
            
    mae_list = np.array(mae_list) #(num_exp, num_seq)
    mape_list = np.array(mape_list)
    rmse_list = np.array(rmse_list)

    preds_list = np.array(preds_list)
    truths_list = np.array(truths_list)
    mean_preds = np.mean(preds_list, axis=0)
    mean_truths = np.mean(truths_list, axis=0)
    outputs = {'prediction': mean_preds, 'truth': mean_truths}
    np.savez_compressed(args.output_filename, **outputs)
    print('Mean predictions saved as {}.'.format(args.output_filename))
   
    seq_len = [2, 4, 8, 16, 24]
    print('--------- PGDENetAQ Final Results ------------' )
    for i, seq in enumerate(seq_len):
        print('Evaluation seq {}:'.format(seq))
        print('MAE | mean: {:.4f} std: {:.4f}'.format(get_mean_std(mae_list[:,i])[0], get_mean_std(mae_list[:,i])[1]))
        print('MAPE | mean: {:.4f} std: {:.4f}'.format(get_mean_std(mape_list[:,i])[0], get_mean_std(mape_list[:,i])[1]))
        print('RMSE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(rmse_list[:,i])[0], get_mean_std(rmse_list[:,i])[1]))
    
    output_text = ''
    output_text += '--------- PGDENetAQ Final Results ------------\n'
    for i, seq in enumerate(seq_len):
        output_text += 'Evaluation seq {}:\n'.format(seq)
        output_text += 'MAE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(mae_list[:,i])[0], get_mean_std(mae_list[:,i])[1])
        output_text += 'MAPE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(mape_list[:,i])[0], get_mean_std(mape_list[:,i])[1])
        output_text += 'RMSE | mean: {:.4f} std: {:.4f}\n\n'.format(get_mean_std(rmse_list[:,i])[0], get_mean_std(rmse_list[:,i])[1])

    # Write the output text to a file
    with open('logs/pgdenet_results.txt', 'w') as file:
        file.write(output_text)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_filename', type=str, default='data/station.csv',help='Sensor Data.')
    parser.add_argument('--config_filename', default='config.yaml', type=str, help='Configuration filename for restoring the model.')
    parser.add_argument('--output_filename', default='models/pgdenet_predictions.npz')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--exp_repeat', type=int, default=5, help='Number of experiments.')
    args, unknown = parser.parse_known_args()
    main(args)