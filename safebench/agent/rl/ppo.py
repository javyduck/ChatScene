''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-01 16:00:55
Description: 
    Copyright (c) 2022-2023 Safebench Team

    Modified from <https://github.com/gouxiangchen/ac-ppo>

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os

import torch
import torch.nn as nn
from fnmatch import fnmatch
from torch.distributions import Normal
import torch.nn.functional as F

from safebench.util.torch_util import CUDA, CPU, hidden_init
from safebench.agent.base_policy import BasePolicy


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        hidden_dim = 64
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.min_val = 1e-8
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_mu.weight.data.uniform_(*hidden_init(self.fc_mu))
        self.fc_std.weight.data.uniform_(*hidden_init(self.fc_std))

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x)) + self.min_val
        return mu, std

    def select_action(self, state, deterministic):
        with torch.no_grad():
            mu, std = self.forward(state)
            if deterministic:
                action = mu
            else:
                n = Normal(mu, std)
                action = n.sample()
        return CPU(action)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        hidden_dim = 64
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO(BasePolicy):
    name = 'PPO'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super(PPO, self).__init__(config, logger)

        self.continue_episode = 0
        self.logger = logger
        self.gamma = config['gamma']
        self.policy_lr = config['policy_lr']
        self.value_lr = config['value_lr']
        self.train_iteration = config['train_iteration']
        self.state_dim = config['ego_state_dim']
        self.action_dim = config['ego_action_dim']
        self.clip_epsilon = config['clip_epsilon']
        self.batch_size = config['batch_size']

        self.model_id = config['model_id']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.policy = CUDA(PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim))
        self.old_policy = CUDA(PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim))
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value = CUDA(ValueNetwork(state_dim=self.state_dim))
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)

        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
            self.old_policy.train()
            self.value.train()
        elif mode == 'eval':
            self.policy.eval()
            self.old_policy.eval()
            self.value.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def get_action(self, state, infos, deterministic=False):
        state_tensor = CUDA(torch.FloatTensor(state))
        action = self.policy.select_action(state_tensor, deterministic)
        return action

    def train(self, replay_buffer):
        self.old_policy.load_state_dict(self.policy.state_dict())

        # start to train, use gradient descent without batch size
        for K in range(self.train_iteration):
            batch = replay_buffer.sample(self.batch_size)
            bn_s = CUDA(torch.FloatTensor(batch['state']))
            bn_a = CUDA(torch.FloatTensor(batch['action']))
            bn_r = CUDA(torch.FloatTensor(batch['reward'])).unsqueeze(-1) # [B, 1]
            bn_s_ = CUDA(torch.FloatTensor(batch['n_state']))
            #bn_d = CUDA(torch.FloatTensor(1-batch['done'])).unsqueeze(-1) # [B, 1]

            with torch.no_grad():
                old_mu, old_std = self.old_policy(bn_s)
                old_n = Normal(old_mu, old_std)
                value_target = bn_r + self.gamma * self.value(bn_s_)
                advantage = value_target - self.value(bn_s)

            # update policy
            mu, std = self.policy(bn_s)
            n = Normal(mu, std)
            log_prob = n.log_prob(bn_a)
            old_log_prob = old_n.log_prob(bn_a)
            ratio = torch.exp(log_prob - old_log_prob)
            L1 = ratio * advantage
            L2 = torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage
            loss = torch.min(L1, L2)
            loss = -loss.mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # update value function
            value_loss = F.mse_loss(value_target, self.value(bn_s))
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()

        # reset buffer
        replay_buffer.reset_buffer()

    def save_model(self, episode):
        states = {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
        }
        filepath = os.path.join(self.model_path, f'model.ppo.{self.model_id}.{episode:04}.torch')
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
        filepath = os.path.join(self.model_path, f'model.ppo.{self.model_id}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading {self.name} model from {filepath}')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No {self.name} model found at {filepath}', 'red')
