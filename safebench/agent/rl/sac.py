''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-01 16:00:31
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os

import torch
import torch.nn as nn
import torch.optim as optim
from fnmatch import fnmatch
from torch.distributions import Normal

from safebench.util.torch_util import CUDA, CPU, kaiming_init
from safebench.agent.base_policy import BasePolicy


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.min_val = 1e-3
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x)) + self.min_val
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, x, a):
        x = x.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((x, a), -1) # combination x and a
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC(BasePolicy):
    name = 'SAC'
    type = 'offpolicy'

    def __init__(self, config, logger):
        self.logger = logger

        self.buffer_start_training = config['buffer_start_training']
        self.lr = config['lr']
        self.continue_episode = 0
        self.state_dim = config['ego_state_dim']
        self.action_dim = config['ego_action_dim']
        self.min_Val = torch.tensor(config['min_Val']).float()
        self.batch_size = config['batch_size']
        self.update_iteration = config['update_iteration']
        self.gamma = config['gamma']
        self.tau = config['tau']

        self.model_id = config['model_id']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # create models
        self.policy_net = CUDA(Actor(self.state_dim, self.action_dim))
        self.value_net = CUDA(Critic(self.state_dim))
        self.Q_net = CUDA(Q(self.state_dim, self.action_dim))
        self.Target_value_net = CUDA(Critic(self.state_dim))

        # create optimizer
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=self.lr)

        # define loss function
        self.value_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()

        # copy parameters
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy_net.train()
            self.value_net.train()
            self.Q_net.train()
        elif mode == 'eval':
            self.policy_net.eval()
            self.value_net.eval()
            self.Q_net.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def get_action(self, state, infos, deterministic=False):
        state = CUDA(torch.FloatTensor(state))
        mu, log_sigma = self.policy_net(state)

        if deterministic:
            action = mu
        else:
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            z = dist.sample()
            action = torch.tanh(z)
        return CPU(action)

    def get_action_log_prob(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        z = dist.sample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + self.min_Val)
        # when action has more than 1 dimensions, we should sum up the log likelihood
        log_prob = torch.sum(log_prob, dim=1, keepdim=True) 
        return action, log_prob, z, batch_mu, batch_log_sigma

    def train(self, replay_buffer):
        if replay_buffer.buffer_len < self.buffer_start_training:
            return

        for _ in range(self.update_iteration):
            # sample replay buffer
            batch = replay_buffer.sample(self.batch_size)
            bn_s = CUDA(torch.FloatTensor(batch['state']))
            bn_a = CUDA(torch.FloatTensor(batch['action']))
            bn_r = CUDA(torch.FloatTensor(batch['reward'])).unsqueeze(-1) # [B, 1]
            bn_s_ = CUDA(torch.FloatTensor(batch['n_state']))
            bn_d = CUDA(torch.FloatTensor(1-batch['done'])).unsqueeze(-1) # [B, 1]

            target_value = self.Target_value_net(bn_s_)
            next_q_value = bn_r + bn_d * self.gamma * target_value

            excepted_value = self.value_net(bn_s)
            excepted_Q = self.Q_net(bn_s, bn_a)

            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(bn_s)
            excepted_new_Q = self.Q_net(bn_s, sample_action)
            next_value = excepted_new_Q - log_prob

            # !!! Note that the actions are sampled according to the current policy, instead of replay buffer. (From original paper)
            V_loss = self.value_criterion(excepted_value, next_value.detach())  # J_V
            V_loss = V_loss.mean()

            # Single Q_net this is different from original paper!!!
            Q_loss = self.Q_criterion(excepted_Q, next_q_value.detach()) # J_Q
            Q_loss = Q_loss.mean()

            log_policy_target = excepted_new_Q - excepted_value
            pi_loss = log_prob * (log_prob - log_policy_target).detach()
            pi_loss = pi_loss.mean()

            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q_optimizer.zero_grad()
            Q_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
            self.Q_optimizer.step()

            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # soft update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

    def save_model(self, episode):
        states = {
            'policy_net': self.policy_net.state_dict(), 
            'value_net': self.value_net.state_dict(), 
            'Q_net': self.Q_net.state_dict()
        }
        filepath = os.path.join(self.model_path, f'model.sac.{self.model_id}.{episode:04}.torch')
        self.logger.log(f'>> Saving {self.name} model to {filepath}')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self, episode=None):
        if episode is None:
            episode = -1
            for _, _, files in os.walk(self.model_path):
                for name in files:
                    if fnmatch(name, "*torch"):
                        cur_episode = int(name.split(".")[-2])
                        if cur_episode > episode:
                            episode = cur_episode
        filepath = os.path.join(self.model_path, f'model.sac.{self.model_id}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading {self.name} model from {filepath}')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.value_net.load_state_dict(checkpoint['value_net'])
            self.Q_net.load_state_dict(checkpoint['Q_net'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No {self.name} model found at {filepath}', 'red')
