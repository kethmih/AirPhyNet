import numpy as np
import torch
import random
import torch.nn as nn
from torch.nn import functional as F

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerParams:
    def __init__(self, rnn_network: nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = nn.Parameter(torch.empty(*shape, device=device))
            nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.empty(length, device=device))
            nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


class GatedFusionModel(torch.nn.Module):
    def __init__(self, num_nodes,latent_dim):
        super(GatedFusionModel, self).__init__()
        self._num_nodes = num_nodes
        self._latent_dim = latent_dim
        self.hid_dim = self._num_nodes*self._latent_dim
        self.fc = nn.Linear(self.hid_dim,self.hid_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, grad_diff, grad_adv):
        X_diff = self.fc(grad_diff)
        X_adv = self.fc(grad_adv)
        z = self.sigmoid(torch.add(X_diff,X_adv))
        H = torch.add((z * X_diff), ((1 - z) * X_adv))
        return H

class ODEFunc(nn.Module):
    def __init__(self, last_wind_vars, num_units, latent_dim, adj_mx, edge_index, edge_attr, gcn_step, num_nodes,
                 gen_layers=1, nonlinearity='tanh', filter_type="diff_adv"):
        """
        :param num_units: dimensionality of the hidden layers
        :param latent_dim: dimensionality used for ODE (input and output). Analog of a continous latent state
        :param adj_mx:
        :param gcn_step:
        :param num_nodes:
        :param gen_layers: hidden layers in each ode func.
        :param nonlinearity:
        """
        super(ODEFunc, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu

        self._num_nodes = num_nodes
        self._num_units = num_units 
        self._latent_dim = latent_dim
        self._gen_layers = gen_layers
        self.nfe = 0 #Number of function integrations
        self.flow_net = LinearNet().to(device)
        self.gated_fusion = GatedFusionModel(self._num_nodes, self._latent_dim)

        self._filter_type = filter_type
        if(self._filter_type == "diff"):
            self._gcn_step = gcn_step
            self._gconv_params = LayerParams(self, 'gconv')
            self._supports = []
            supports = []
            supports.append(calculate_scaled_laplacian(adj_mx))

            for support in supports:
                self._supports.append(self._build_sparse_matrix(support))

        elif(self._filter_type == "adv"):
            edge_index = torch.LongTensor(edge_index).to(device) 
            edge_attr = torch.Tensor(np.float32(edge_attr)).to(device) 
            edge_src, edge_target = edge_index

            last_wind_vars = self.flow_net(last_wind_vars) 
            node_src = last_wind_vars[:,edge_src] 
            node_target = last_wind_vars[:,edge_target]
            edge_weight = node_src - node_target
            edge_weight = edge_weight.squeeze()

            batch_size, num_edges =  edge_weight.shape
            adj_mx_adv = torch.zeros(batch_size, num_nodes, num_nodes)
            for batch_index in range(batch_size):
                for edge_id in range(num_edges):
                    src_node = edge_index[0, edge_id]
                    tgt_node = edge_index[1, edge_id]

                    # Assign the edge weight to the adjacency matrix
                    adj_mx_adv[batch_index,src_node, tgt_node] = edge_weight[batch_index,edge_id]

            self._gcn_step = gcn_step
            self._gconv_adv_params = LayerParams(self, 'gconv_adv')

            #For Advection
            self._supports_adv = []
            supports_adv = []
            for i in range(batch_size):
                adj_mx_new  = adj_mx_adv[i]
                supports_adv.append(calculate_scaled_laplacian(adj_mx_new.detach().numpy()))
            for support in supports_adv:
                self._supports_adv.append(self._build_sparse_matrix(support))

        elif(self._filter_type == "diff_adv"):
            edge_index = torch.LongTensor(edge_index).to(device) 
            edge_attr = torch.Tensor(np.float32(edge_attr)).to(device) 
            edge_src, edge_target = edge_index

            last_wind_vars = self.flow_net(last_wind_vars) 
            node_src = last_wind_vars[:,edge_src]
            node_target = last_wind_vars[:,edge_target]
            edge_weight = node_src - node_target
            edge_weight = edge_weight.squeeze() 

            batch_size, num_edges =  edge_weight.shape
            adj_mx_adv = torch.zeros(batch_size, num_nodes, num_nodes)
            for batch_index in range(batch_size):
                for edge_id in range(num_edges):
                    src_node = edge_index[0, edge_id]
                    tgt_node = edge_index[1, edge_id]

                    # Assign the edge weight to the adjacency matrix
                    adj_mx_adv[batch_index,src_node, tgt_node] = edge_weight[batch_index,edge_id]

            self._gcn_step = gcn_step
            self._gconv_params = LayerParams(self, 'gconv')
            self._gconv_adv_params = LayerParams(self, 'gconv_adv')

            #For Advection
            self._supports_adv = []
            supports_adv = []
            for i in range(batch_size):
                adj_mx_new  = adj_mx_adv[i]
                supports_adv.append(calculate_scaled_laplacian(adj_mx_new.detach().numpy()))
            for support in supports_adv:
                self._supports_adv.append(self._build_sparse_matrix(support))

            #For Diffusion
            self._supports = []
            supports = []
            supports.append(calculate_scaled_laplacian(adj_mx))
            for support in supports:
                self._supports.append(self._build_sparse_matrix(support))
            
        elif(self._filter_type == "unkP"):
            ode_func_net = create_net(latent_dim, latent_dim, n_units = num_units)
            init_network_weights(ode_func_net)
            self.gradient_net = ode_func_net
            
        else:
            print("Invalid Filter Type")


    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def forward(self, t_local, y, backwards = False):
        """
        t_local: current time point
        y: value at the current time point, shape (B, num_nodes * latent_dim)
        Output: (B, num_nodes * latent_dim)`.
        """
        self.nfe += 1
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, inputs):
        coeff = 0.1
        if(self._filter_type == "diff"):
            grad = - coeff * self.ode_func_net_diff(inputs, self._supports)
        elif(self._filter_type == "adv"):
            grad = - self.ode_func_net_adv(inputs, self._supports_adv)
        elif(self._filter_type == "diff_adv"):
            grad_diff = - coeff * self.ode_func_net_diff(inputs, self._supports)
            grad_adv = - self.ode_func_net_adv(inputs, self._supports_adv)
            grad = self.gated_fusion(grad_diff, grad_adv)
        elif(self._filter_type == "unkP"):
            grad = self._fc(inputs)
        else:
            print("Invalid Filter Type")

        return grad

    def ode_func_net_diff(self, inputs, _supports):
        c = inputs
        for i in range(self._gen_layers):
            c = self._gconv_dif(c, self._num_units,_supports)
            c = self._activation(c)
        c = self._gconv_dif(c, self._latent_dim,_supports)
        c = self._activation(c)
        return c

    def ode_func_net_adv(self, inputs,_supports_adv):
        c = inputs
        for i in range(self._gen_layers):
            c = self._gconv_adv(c, self._num_units,_supports_adv)
            c = self._activation(c)
        c = self._gconv_adv(c, self._latent_dim,_supports_adv)
        c = self._activation(c)
        return c


    def _fc(self, inputs):
        batch_size = inputs.size()[0]
        grad = self.gradient_net(inputs.view(batch_size * self._num_nodes, self._latent_dim))
        return grad.reshape(batch_size, self._num_nodes * self._latent_dim) # (batch_size, num_nodes, latent_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _gconv_dif(self, inputs, output_size,_supports, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        input_size = inputs.size(2)

        x = inputs
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._gcn_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._gcn_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._gcn_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    def _gconv_adv(self, inputs, output_size,_supports_adv, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        input_size = inputs.size(2)

        x = inputs
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._gcn_step == 0:
            pass
        else:
            for support in self._supports_adv:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._gcn_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports_adv) * self._gcn_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_adv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_adv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])