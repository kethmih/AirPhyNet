import torch
import torch.nn as nn
import time

from torchdiffeq import odeint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffeqSolver(nn.Module):
    def __init__(self, odefunc, method, latent_dim,
            odeint_rtol = 1e-4, odeint_atol = 1e-5):
        nn.Module.__init__(self)

        self.ode_method = method
        self.odefunc = odefunc
        self.latent_dim = latent_dim

        self.rtol = odeint_rtol
        self.atol = odeint_atol

    def forward(self, first_point, time_steps_to_pred):
        """
        Decoder the trajectory through the ODE Solver.
        :param time_steps_to_pred: horizon
        :param first_point: (n_traj_samples, batch_size, num_nodes * latent_dim)
        :return: pred_y: # shape (horizon, n_traj_samples, batch_size, self.num_nodes * self.output_dim)
        """
        n_traj_samples, batch_size = first_point.size()[0], first_point.size()[1]
        first_point = first_point.reshape(n_traj_samples * batch_size, -1) # reduce the complexity by merging dimension

        # pred_y shape: (horizon, n_traj_samples * batch_size, num_nodes * latent_dim)
        start_time = time.time()
        self.odefunc.nfe = 0
        pred_y = odeint(self.odefunc,
                            first_point,
                            time_steps_to_pred,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.ode_method)
        time_fe = time.time() - start_time

        # pred_y shape: (horizon, n_traj_samples, batch_size, num_nodes * latent_dim)
        pred_y = pred_y.reshape(pred_y.size()[0], n_traj_samples, batch_size, -1)

        return pred_y, (self.odefunc.nfe, time_fe)