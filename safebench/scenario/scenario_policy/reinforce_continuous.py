''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-22 17:01:35
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>

    This file implements the method proposed in paper:
        Learning to Collide: An Adaptive Safety-Critical Scenarios Generating Method
        <https://arxiv.org/pdf/2003.01197.pdf>
'''

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

from safebench.scenario.scenario_policy.base_policy import BasePolicy
from safebench.util.torch_util import CUDA, CPU


def normalize_routes(routes):
    mean_x = np.mean(routes[:, 0:1])
    max_x = np.max(np.abs(routes[:, 0:1]))
    x_1_2 = (routes[:, 0:1] - mean_x) / max_x

    mean_y = np.mean(routes[:, 1:2])
    max_y = np.max(np.abs(routes[:, 1:2]))
    y_1_2 = (routes[:, 1:2] - mean_y) / max_y

    route = np.concatenate([x_1_2, y_1_2], axis=0)
    return route


class IndependantModel(nn.Module):
    def __init__(self, num_waypoint=20):
        super(IndependantModel, self).__init__()
        input_size = num_waypoint*2 + 1
        hidden_size_1 = 64

        self.a_os = 1
        self.b_os = 1
        self.c_os = 1
        self.d_os = 1

        self.relu = nn.ReLU()
        self.fc_input = nn.Sequential(nn.Linear(input_size, hidden_size_1))
        self.fc_action_a = nn.Sequential(nn.Linear(hidden_size_1, self.a_os*2))
        self.fc_action_b = nn.Sequential(nn.Linear(1+hidden_size_1, self.b_os*2))
        self.fc_action_c = nn.Sequential(nn.Linear(1+1+hidden_size_1, self.c_os*2))
        self.fc_action_d = nn.Sequential(nn.Linear(1+1+1+hidden_size_1, self.d_os*2))

    def sample_action(self, normal_action, action_os):
        # get the mu and sigma
        mu = normal_action[:, :action_os]
        sigma = F.softplus(normal_action[:, action_os:])

        # calculate the probability by mu and sigma of normal distribution
        eps = CUDA(torch.randn(mu.size()))
        action = (mu + sigma*eps)
        return action, mu, sigma

    def forward(self, x, determinstic):
        # p(s)
        s = self.fc_input(x)
        s = self.relu(s)

        # p(a|s)
        normal_a = self.fc_action_a(s)
        action_a, mu_a, sigma_a = self.sample_action(normal_a, self.a_os)

        # p(b|a,s) 
        normal_b = self.fc_action_b(s)
        action_b, mu_b, sigma_b = self.sample_action(normal_b, self.b_os)

        # p(c|a,b,s)
        normal_c = self.fc_action_c(s)
        action_c, mu_c, sigma_c = self.sample_action(normal_c, self.c_os)

        # p(d|a,b,c,s)
        normal_d = self.fc_action_d(s)
        action_d, mu_d, sigma_d = self.sample_action(normal_d, self.d_os)

        # concate
        action = torch.cat((action_a, action_b, action_c, action_d), dim=1) # [B, 4]
        mu = torch.cat((mu_a, mu_b, mu_c, mu_d), dim=1)                     # [B, 4]
        sigma = torch.cat((sigma_a, sigma_b, sigma_c, sigma_d), dim=1)      # [B, 4]
        return mu, sigma, action


class AutoregressiveModel(nn.Module):
    def __init__(self, num_waypoint=30, standard_action_dim=True):
        super(AutoregressiveModel, self).__init__()
        self.standard_action_dim = standard_action_dim
        input_size = num_waypoint*2 + 1
        hidden_size_1 = 32

        self.a_os = 1
        self.b_os = 1
        self.c_os = 1
        if self.standard_action_dim:
            self.d_os = 1

        self.relu = nn.ReLU()
        self.fc_input = nn.Sequential(nn.Linear(input_size, hidden_size_1))
        self.fc_action_a = nn.Sequential(nn.Linear(hidden_size_1, self.a_os*2))
        self.fc_action_b = nn.Sequential(nn.Linear(1+hidden_size_1, self.b_os*2))
        self.fc_action_c = nn.Sequential(nn.Linear(1+1+hidden_size_1, self.c_os*2))
        if self.standard_action_dim:
            self.fc_action_d = nn.Sequential(nn.Linear(1+1+1+hidden_size_1, self.d_os*2))

    def sample_action(self, normal_action, action_os):
        # get the mu and sigma
        mu = normal_action[:, :action_os]
        sigma = F.softplus(normal_action[:, action_os:])

        # calculate the probability by mu and sigma of normal distribution
        eps = CUDA(torch.randn(mu.size()))
        action = mu + sigma * eps
        return action, mu, sigma

    def forward(self, x, determinstic):
        # p(s)
        s = self.fc_input(x)
        s = self.relu(s)

        # p(a|s)
        normal_a = self.fc_action_a(s)
        action_a, mu_a, sigma_a = self.sample_action(normal_a, self.a_os)

        # p(b|a,s)
        state_sample_a = torch.cat((s, mu_a), dim=1) if determinstic else torch.cat((s, action_a), dim=1) 
        normal_b = self.fc_action_b(state_sample_a)
        action_b, mu_b, sigma_b = self.sample_action(normal_b, self.b_os)

        # p(c|a,b,s)
        state_sample_a_b = torch.cat((s, mu_a, mu_b), dim=1) if determinstic else torch.cat((s, action_a, action_b), dim=1)
        normal_c = self.fc_action_c(state_sample_a_b)
        action_c, mu_c, sigma_c = self.sample_action(normal_c, self.c_os)

        # p(d|a,b,c,s)
        if self.standard_action_dim:
            state_sample_a_b_c = torch.cat((s, mu_a, mu_b, mu_c), dim=1) if determinstic else torch.cat((s, action_a, action_b, action_c), dim=1)
            normal_d = self.fc_action_d(state_sample_a_b_c)
            action_d, mu_d, sigma_d = self.sample_action(normal_d, self.d_os)

        # concate
        if self.standard_action_dim:
            action = torch.cat((action_a, action_b, action_c, action_d), dim=1) # [B, 4]
            mu = torch.cat((mu_a, mu_b, mu_c, mu_d), dim=1)                     # [B, 4]
            sigma = torch.cat((sigma_a, sigma_b, sigma_c, sigma_d), dim=1)      # [B, 4]
        else:
            action = torch.cat((action_a, action_b, action_c), dim=1)           # [B, 3]
            mu = torch.cat((mu_a, mu_b, mu_c), dim=1)                           # [B, 3]
            sigma = torch.cat((sigma_a, sigma_b, sigma_c), dim=1)               # [B, 3]
        return mu, sigma, action


class REINFORCE(BasePolicy):
    name = 'reinforce'
    type = 'init_state'

    def __init__(self, scenario_config, logger):
        self.logger = logger
        self.num_waypoint = 30
        self.continue_episode = 0
        self.num_scenario = scenario_config['num_scenario']
        self.batch_size = scenario_config['batch_size']
        self.model_path = os.path.join(scenario_config['ROOT_DIR'], scenario_config['model_path'])
        self.model_id = scenario_config['model_id']
        self.lr = scenario_config['lr']
        self.entropy_weight = 0.0001

        self.standard_action_dim = True
        self.model = CUDA(AutoregressiveModel(self.num_waypoint))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, replay_buffer):
        if replay_buffer.init_buffer_len < self.batch_size:
            return

        # get episode reward
        batch = replay_buffer.sample_init(self.batch_size)
        episode_reward = batch['episode_reward']
        log_prob = batch['log_prob']
        entropy = batch['entropy']
        
        episode_reward = CUDA(torch.tensor(episode_reward, dtype=torch.float32))
        episode_reward = -episode_reward # objective is to minimize the reward

        # we only have one step
        loss = log_prob * episode_reward - entropy * self.entropy_weight
        loss = loss.mean(dim=0)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logger.log('>> Training loss: {:.4f}'.format(loss.item()))

        # reset the buffer since this is a on-policy method
        replay_buffer.reset_init_buffer()

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def proceess_init_state(self, state):
        processed_state_list = []
        for s_i in range(len(state)):
            route = state[s_i]['route']
            target_speed = state[s_i]['target_speed'] / 10.0

            index = np.linspace(1, len(route) - 1, self.num_waypoint).tolist()
            index = [int(i) for i in index]
            route_norm = normalize_routes(route[index])[:, 0] # [num_waypoint*2]
            processed_state = np.concatenate((route_norm, [target_speed]), axis=0).astype('float32')
            processed_state_list.append(processed_state)
        
        processed_state_list = np.stack(processed_state_list, axis=0)
        return processed_state_list

    def get_action(self, state, infos, deterministic=False):
        return [None] * self.num_scenario

    def get_init_action(self, state, deterministic=False):
        # the state should be a sequence of route waypoints
        processed_state = self.proceess_init_state(state)
        processed_state = CUDA(torch.from_numpy(processed_state))

        mu, sigma, action = self.model.forward(processed_state, deterministic)

        # calculate the probability that this distribution outputs this action
        action_dist = Normal(mu, sigma)
        log_prob = action_dist.log_prob(action).sum(dim=1) # [B]

        # calculate the entropy
        action_entropy = 0.5*(2 * np.pi * sigma**2).log() + 0.5
        entropy = action_entropy.sum(dim=1) # [B]

        # clip the action to [-1, 1]
        action = np.clip(CPU(action), -1.0, 1.0)
        additional_info = {'log_prob': log_prob, 'entropy': entropy}
        return action, additional_info

    def load_model(self, scenario_configs=None):
        assert scenario_configs is not None, 'Scenario configs should be provided for loading model.'
        scenario_id = scenario_configs[0].scenario_id
        model_file = scenario_configs[0].parameters[0]
        self.standard_action_dim = scenario_configs[0].parameters[1]
        for config in scenario_configs:
            assert scenario_id == config.scenario_id, 'Scenarios should be the same in a batch.'
            assert model_file == config.parameters[0], 'Model filenames should be the same in a batch.'
            assert self.standard_action_dim == config.parameters[1], 'Action dimensions should be the same in a batch.'

        # TODO: remove this after obtaining new models with consistent action dim
        self.model = CUDA(AutoregressiveModel(self.num_waypoint, self.standard_action_dim))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        model_filename = os.path.join(self.model_path, str(scenario_id), model_file)
        if os.path.exists(model_filename):
            self.logger.log(f'>> Loading lc model from {model_filename}')
            with open(model_filename, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['parameters'])
        else:
            self.logger.log(f'>> Fail to find lc model from {model_filename}', color='yellow')

    def save_model(self, epoch):
        if not os.path.exists(self.model_path):
            self.logger.log(f'>> Creating folder for saving model: {self.model_path}')
            os.makedirs(self.model_path)
        model_filename = os.path.join(self.model_path, f'{self.model_id}.pt')
        self.logger.log(f'>> Saving lc model to {model_filename}')
        with open(model_filename, 'wb+') as f:
            torch.save({'parameters': self.model.state_dict()}, f)
