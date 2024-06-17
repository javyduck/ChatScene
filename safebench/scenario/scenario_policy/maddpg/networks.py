''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:46:28
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients>

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# "normal" or "uniform" or None
INIT_METHOD = "normal"

def mlp(sizes, activation, output_activation=nn.Identity):
    if INIT_METHOD == "normal":
        initializer = nn.init.xavier_normal_
    elif INIT_METHOD == "uniform":
        initializer = nn.init.xavier_uniform_
    else:
        initializer = None
    bias_init = 0.0
    in_fn = nn.BatchNorm1d(sizes[0])
    in_fn.weight.data.fill_(1)
    in_fn.bias.data.fill_(0)
    layers = [in_fn]
#     layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
            # init layer weight
        if j < len(sizes) - 2:
            if initializer is not None:
                initializer(layer.weight)
                nn.init.constant_(layer.bias, bias_init)
        else:
            layer.weight.data.uniform_(-2e-3,2e-3)
            nn.init.constant_(layer.bias, bias_init)
        layers += [layer, act()]
    return nn.Sequential(*layers)

class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit=1):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)
    
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir = './checkpoints/'):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.actor = MLPActor(input_dims, n_actions, [fc1_dims, fc2_dims], nn.ReLU)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = self.actor(state)
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir = './checkpoints/', num_q=2):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.critic = nn.ModuleList([
            mlp([input_dims + n_agents * n_actions] + list([fc1_dims, fc2_dims]) + [1], nn.ReLU)
            for i in range(num_q)
        ])

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)
        
    def forward(self, obs, act):
        # Squeeze is critical to ensure value has the right shape.
        # Without squeeze, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        data = torch.cat([obs, act], dim=-1)
        return [torch.squeeze(q(data), -1) for q in self.critic]

    def predict(self, obs, act):
        q_list = self.forward(obs, act)
        qs = torch.vstack(q_list)  # [num_q, batch_size]
        return torch.min(qs, dim=0).values, q_list

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

        
