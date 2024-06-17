import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from fnmatch import fnmatch

from safebench.util.torch_util import CUDA, CPU
from safebench.agent.base_policy import BasePolicy


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.network(x)
        x = self.tanh(x)
        return x


class DoubleQFunc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class TD3(BasePolicy):
    name = 'TD3'
    type = 'offpolicy'

    def __init__(self, config, logger):
        self.logger = logger

        self.buffer_start_training = config['buffer_start_training']
        self.lr = config['lr']
        self.continue_episode = 0
        self.state_dim = config['ego_state_dim']
        self.action_dim = config['ego_action_dim']
        self.batch_size = config['batch_size']
        self.hidden_size = config['hidden_size']
        self.update_iteration = config['update_iteration']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.update_interval = config['update_interval']
        self.action_lim = config['action_lim']
        self.target_noise = config['target_noise']
        self.target_noise_clip = config['target_noise_clip']
        self.explore_noise = config['explore_noise']

        self.model_id = config['model_id']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # aka critic
        self.q_funcs = CUDA(DoubleQFunc(self.state_dim, self.action_dim, hidden_size=self.hidden_size))
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = CUDA(Policy(self.state_dim, self.action_dim, hidden_size=self.hidden_size))
        self.target_policy = copy.deepcopy(self.policy)
        for p in self.target_policy.parameters():
            p.requires_grad = False

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=self.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self._update_counter = 0
        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.q_funcs.train()
            self.target_q_funcs.train()
            self.policy.train()
            self.target_policy.train()
        elif mode == 'eval':
            self.q_funcs.eval()
            self.target_q_funcs.eval()
            self.policy.eval()
            self.target_policy.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def get_action(self, state, infos, deterministic=False):
        state = CUDA(torch.FloatTensor(state))
        with torch.no_grad():
            action = self.policy(state)
        if not deterministic:
            action += self.explore_noise * torch.randn_like(action)
        action.clamp_(-self.action_lim, self.action_lim)
        return CPU(action)

    def update_target(self):
        # moving average update of target networks
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
            for target_pi_param, pi_param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_pi_param.data.copy_(self.tau * pi_param.data + (1.0 - self.tau) * target_pi_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch = self.target_policy(nextstate_batch)
            target_noise = self.target_noise * torch.randn_like(nextaction_batch)
            target_noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction_batch += target_noise
            nextaction_batch.clamp_(-self.action_lim, self.action_lim)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy(self, state_batch):
        action_batch = self.policy(state_batch)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (-qval_batch).mean()
        return policy_loss

    def train(self, replay_buffer):
        if replay_buffer.buffer_len < self.buffer_start_training:
            return

        q1_loss, q2_loss, pi_loss = 0, 0, None

        for _ in range(self.update_iteration):
            # sample replay buffer
            batch = replay_buffer.sample(self.batch_size)
            state_batch = CUDA(torch.FloatTensor(batch['state']))
            nextstate_batch = CUDA(torch.FloatTensor(batch['n_state']))
            action_batch = CUDA(torch.FloatTensor(batch['action']))
            reward_batch = CUDA(torch.FloatTensor(batch['reward'])).unsqueeze(-1) # [B, 1]
            done_batch = CUDA(torch.FloatTensor(1-batch['done'])).unsqueeze(-1) # [B, 1]

            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_functions(
                state_batch, 
                action_batch, 
                reward_batch,
                nextstate_batch, 
                done_batch
            )
            q_loss_step = q1_loss_step + q2_loss_step
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()

            self._update_counter += 1

            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()

            if self._update_counter % self.update_interval == 0:
                if not pi_loss:
                    pi_loss = 0
                # update policy
                for p in self.q_funcs.parameters():
                    p.requires_grad = False
                pi_loss_step = self.update_policy(state_batch)
                self.policy_optimizer.zero_grad()
                pi_loss_step.backward()
                self.policy_optimizer.step()
                for p in self.q_funcs.parameters():
                    p.requires_grad = True
                # update target policy and q-functions using Polyak averaging
                self.update_target()
                pi_loss += pi_loss_step.detach().item()

        return q1_loss, q2_loss, pi_loss

    def save_model(self, episode):
        states = {
            'q_funcs': self.q_funcs.state_dict(),
            'target_q_funcs': self.target_q_funcs.state_dict(),
            'policy': self.policy.state_dict(),
            'target_policy': self.target_policy.state_dict(),
        }
        filepath = os.path.join(self.model_path, f'model.td3.{self.model_id}.{episode:04}.torch')
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
        filepath = os.path.join(self.model_path, f'model.td3.{self.model_id}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading {self.name} model from {filepath}')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.q_funcs.load_state_dict(checkpoint['q_funcs'])
            self.target_q_funcs.load_state_dict(checkpoint['target_q_funcs'])
            self.policy.load_state_dict(checkpoint['policy'])
            self.target_policy.load_state_dict(checkpoint['target_policy'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No {self.name} model found at {filepath}', 'red')
