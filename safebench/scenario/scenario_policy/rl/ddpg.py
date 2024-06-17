''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 22:17:07
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os
import numpy as np

import torch
import torch.nn as nn
from fnmatch import fnmatch
import torch.nn.functional as F
import torch.optim as optim

from safebench.util.torch_util import CUDA, CPU, hidden_init
from safebench.agent.base_policy import BasePolicy


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        l1_size = 128
        l2_size = 128
        self.l1 = nn.Linear(state_dim, l1_size)
        self.l2 = nn.Linear(l1_size, l2_size)
        self.l3 = nn.Linear(l2_size, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.l1.weight.data.uniform_(*hidden_init(self.l1))
        self.l2.weight.data.uniform_(*hidden_init(self.l2))
        self.l3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        l1_size = 128
        l2_size = 128
        self.l1 = nn.Linear(state_dim, l1_size)
        self.l2 = nn.Linear(l1_size+action_dim, l2_size)
        self.l3 = nn.Linear(l2_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.l1.weight.data.uniform_(*hidden_init(self.l1))
        self.l2.weight.data.uniform_(*hidden_init(self.l2))
        self.l3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, u):
        xs = F.relu(self.l1(x))
        x = torch.cat([xs, u], dim=1)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(BasePolicy):
    name = 'DDPG'
    type = 'offpolicy'

    def __init__(self, config, logger):
        self.logger = logger

        self.continue_episode = 0
        self.state_dim = config['ego_state_dim']
        self.action_dim = config['ego_action_dim']
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.tau = config['tau']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.update_iteration = config['update_iteration']
        self.buffer_start_training = config['buffer_start_training']
        self.epsilon = config['epsilon']

        self.model_id = config['model_id']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.actor = CUDA(Actor(self.state_dim, self.action_dim))
        self.actor_target = CUDA(Actor(self.state_dim, self.action_dim))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = CUDA(Critic(self.state_dim, self.action_dim))
        self.critic_target = CUDA(Critic(self.state_dim, self.action_dim))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.actor.train()
            self.actor_target.train()
            self.critic.train()
            self.critic_target.train()
        elif mode == 'eval':
            self.actor.eval()
            self.actor_target.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')
    
    def info_process(self, infos):
        info_batch = np.stack([i_i['actor_info'] for i_i in infos], axis=0)
        info_batch = info_batch.reshape(info_batch.shape[0], -1)
        return info_batch

    def get_init_action(self, state, deterministic=False):
        num_scenario = len(state)
        additional_in = {}
        return [None] * num_scenario, additional_in

    def get_action(self, state, infos, deterministic=False):
        if np.random.randn() > self.epsilon or deterministic: # greedy policy
            state = self.info_process(infos)
            state = CUDA(torch.FloatTensor(state))
            action = self.actor(state).cpu().data.numpy()
        else: # random policy
            action = np.random.uniform(-1.0, 1.0, size=(state.shape[0], self.action_dim))
        
        # decay epsilon
        self.epsilon *= 0.99

        return action

    def train(self, replay_buffer):
        # check if memory is enough for one batch
        if replay_buffer.buffer_len < self.buffer_start_training:
            return

        for it in range(self.update_iteration):
            # sample replay buffer
            batch = replay_buffer.sample(self.batch_size)
            bn_s = CUDA(torch.FloatTensor(batch['actor_info'])).reshape(self.batch_size, -1)
            bn_s_ = CUDA(torch.FloatTensor(batch['n_actor_info'])).reshape(self.batch_size, -1)
            bn_a = CUDA(torch.FloatTensor(batch['action']))
            bn_r = CUDA(torch.FloatTensor(batch['reward'])).unsqueeze(-1) # [B, 1]
            bn_d = CUDA(torch.FloatTensor(1-batch['done'])).unsqueeze(-1) # [B, 1]

            # Compute the target Q value
            target_Q = self.critic_target(bn_s_, self.actor_target(bn_s_))
            target_Q = bn_r + (bn_d * self.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(bn_s, bn_a)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(bn_s, self.actor(bn_s)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, episode):
        states = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }
        filepath = os.path.join(self.model_path, f'model.ddpg.{self.model_id}.{episode:04}.torch')
        self.logger.log(f'>> Saving scenario policy {self.name} model to {filepath}')
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
        filepath = os.path.join(self.model_path, f'model.ddpg.{self.model_id}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading scenario policy {self.name} model from {filepath}')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No scenario policy {self.name} model found at {filepath}', 'red')
            exit()