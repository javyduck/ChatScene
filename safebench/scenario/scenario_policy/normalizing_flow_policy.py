''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-22 17:26:29
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>

    This file implements the method proposed in paper:
        Multimodal Safety-Critical Scenarios Generation for Decision-Making Algorithms Evaluation
        <https://arxiv.org/pdf/2009.08311.pdf>
'''

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

from safebench.scenario.scenario_policy.reinforce_continuous import REINFORCE
from safebench.util.torch_util import CUDA, CPU


class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3_s = nn.Linear(n_hidden, n_output)
        self.fc3_t = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        hidden = F.relu(self.fc2(F.relu(self.fc1(x))))
        s = torch.tanh(self.fc3_s(hidden))
        t = self.fc3_t(hidden)
        return s, t


class ConditionalRealNVP(nn.Module):
    def __init__(self, n_flows, condition_dim, data_dim, n_hidden):
        super(ConditionalRealNVP, self).__init__()
        self.n_flows = n_flows
        self.condition_dim = condition_dim

        # divide the data dimension by 1/2 to do the affine operation
        assert(data_dim % 2 == 0)
        self.n_half = int(data_dim/2)

        # build the network list
        self.NN = torch.nn.ModuleList()
        for k in range(n_flows):
            # the input of each layer should also contain the condition
            self.NN.append(MLP(self.n_half+self.condition_dim, self.n_half, n_hidden))
        
    def forward(self, x, c):
        log_det_jacobian = 0
        for k in range(self.n_flows):
            x_a = x[:, :self.n_half]
            x_b = x[:, self.n_half:]
            
            x_a_c = torch.cat([x_a, c], dim=1)
            s, t = self.NN[k](x_a_c)
            x_b = torch.exp(s)*x_b + t
            
            x = torch.cat([x_b, x_a], dim=1)
            log_det_jacobian += s
        
        return x, log_det_jacobian
        
    def inverse(self, z, c):
        for k in reversed(range(self.n_flows)):
            z_a = z[:, self.n_half:]
            z_b = z[:, :self.n_half]

            z_a_c = torch.cat([z_a, c], dim=1)
            s, t = self.NN[k](z_a_c)
            z_b = (z_b - t) / torch.exp(s)
            z = torch.cat([z_a, z_b], dim=1)
        return z


# for prior model
class RealNVP(nn.Module):
    def __init__(self, n_flows, data_dim, n_hidden):
        super(RealNVP, self).__init__()
        self.n_flows = n_flows

        # divide the data dimension by 1/2 to do the affine operation
        assert(data_dim % 2 == 0)
        self.n_half = int(data_dim/2)

        # build the network list
        self.NN = torch.nn.ModuleList()
        for k in range(n_flows):
            # the input of each layer should also contain the condition
            self.NN.append(MLP(self.n_half, self.n_half, n_hidden))
        
    def forward(self, x):
        log_det_jacobian = 0
        for k in range(self.n_flows):
            x_a = x[:, :self.n_half]
            x_b = x[:, self.n_half:]

            s, t = self.NN[k](x_a)
            x_b = torch.exp(s)*x_b + t
            x = torch.cat([x_b, x_a], dim=1)
            log_det_jacobian += s
        
        return x, log_det_jacobian
        
    def inverse(self, z):
        for k in reversed(range(self.n_flows)):
            z_a = z[:, self.n_half:]
            z_b = z[:, :self.n_half]
            s, t = self.NN[k](z_a)
            z_b = (z_b - t) / torch.exp(s)
            z = torch.cat([z_a, z_b], dim=1)
        return z


class NormalizingFlow(REINFORCE):
    name = 'nf'
    type = 'init_state'

    def __init__(self, scenario_config, logger):
        self.logger = logger
        self.num_waypoint = 30
        self.continue_episode = 0
        self.num_scenario = scenario_config['num_scenario']
        self.model_path = os.path.join(scenario_config['ROOT_DIR'], scenario_config['model_path'])
        self.model_id = scenario_config['model_id']
        self.use_prior = scenario_config['use_prior']

        self.lr = scenario_config['lr']
        self.batch_size = scenario_config['batch_size']
        self.prior_lr = scenario_config['prior_lr']

        self.prior_epochs = scenario_config['prior_epochs']
        self.alpha = scenario_config['alpha']
        self.itr_per_train = scenario_config['itr_per_train']

        self.state_dim = scenario_config['state_dim']
        self.action_dim = scenario_config['action_dim']
        self.reward_dim = scenario_config['reward_dim']
        self.drop_threshold = scenario_config['drop_threshold']
        self.n_flows = scenario_config['n_flows_model']

        # latent space
        self.z = MultivariateNormal(CUDA(torch.zeros(self.action_dim)), CUDA(torch.eye(self.action_dim)))

        # prior model and generator
        self.prior_model = CUDA(RealNVP(n_flows=self.n_flows, data_dim=self.action_dim, n_hidden=128))
        self.model = CUDA(ConditionalRealNVP(n_flows=self.n_flows, condition_dim=self.state_dim, data_dim=self.action_dim, n_hidden=64))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_prior_model(self, prior_data):
        """ 
            Train the prior model using the data from the prior distribution.
            This function should be used seperately from the Safebench framework to train the prior model.
        """
        prior_data = CUDA(torch.tensor(prior_data))
        # papre a data loader
        train_loader = torch.utils.data.DataLoader(prior_data, shuffle=True, batch_size=self.batch_size)
        self.prior_optimizer = optim.Adam(self.prior_model.parameters(), lr=self.prior_lr)
        self.prior_model.train()

        # train the model
        for epoch in range(self.prior_epochs):
            avg_loglikelihood = []
            for data in train_loader:
                sample_z, log_det_jacobian = self.prior_model(data)
                log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdims=True)
                loglikelihood = -torch.mean(self.z.log_prob(sample_z)[:, None] + log_det_jacobian)
                self.prior_optimizer.zero_grad()
                loglikelihood.backward()
                self.prior_optimizer.step()
                avg_loglikelihood.append(loglikelihood.item())
            self.logger.log('[{}/{}] Prior training error: {}'.format(epoch, self.prior_epochs, np.mean(avg_loglikelihood)))

    def prior_likelihood(self, actions):
        sample_z, log_det_jacobian = self.prior_model(actions)
        log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdims=True)
        loglikelihood = self.z.log_prob(sample_z)[:, None] + log_det_jacobian
        prob = torch.exp(loglikelihood)
        return prob

    def flow_likelihood(self, actions, condition):
        sample_z, log_det_jacobian = self.model(actions, condition)
        # make sure the dimension is aligned, for action_dim > 2, the log_det is more than 1 dimension
        log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdims=True)
        loglikelihood = self.z.log_prob(sample_z)[:, None] + log_det_jacobian
        return loglikelihood

    def prior_sample(self, sample_number=1000, sigma=1.0):
        sampler = MultivariateNormal(CUDA(torch.zeros(self.action_dim)), CUDA(sigma*torch.eye(self.action_dim)))
        new_sampled_z = sampler.sample((sample_number,))

        self.prior_model.eval()
        with torch.no_grad():
            prior_flow = self.prior_model.inverse(new_sampled_z)
        return prior_flow.cpu().numpy()

    def flow_sample(self, state, sample_number=1000, sigma=1.0):
        # use a new sampler, then we can control the sigma 
        sampler = MultivariateNormal(CUDA(torch.zeros(self.action_dim)), CUDA(sigma*torch.eye(self.action_dim)))
        new_sampled_z = sampler.sample((sample_number,))

        # condition should be repeated sample_number times
        condition = CUDA(torch.tensor(state))[None]
        condition = condition.repeat(sample_number, 1)

        self.model.eval()
        with torch.no_grad():
            action_flow = self.model.inverse(new_sampled_z, condition)
        return CPU(action_flow)

    def get_init_action(self, state, infos, deterministic=False):
        # the state should be a sequence of route waypoints
        processed_state = self.proceess_init_state(state)
        processed_state = CUDA(torch.from_numpy(processed_state))

        self.model.eval()
        with torch.no_grad():
            mean = CUDA(torch.zeros(self.action_dim))[None]
            condition = CUDA(torch.tensor(processed_state))[None]
            action = self.model.inverse(mean, condition)

        action_list = []
        for a_i in range(self.action_dim):
            action_list.append(action.cpu().numpy()[0, a_i])
        return action_list

    # train on batched data
    def train(self, replay_buffer):
        if replay_buffer.init_buffer_len < self.batch_size:
            return

        self.model.train()
        # the buffer can be resued since we evaluate action-state every time
        for _ in range(self.itr_per_train):
            # get episode reward
            batch = replay_buffer.sample_init(self.batch_size)
            state = batch['static_obs']
            action = batch['init_action']
            episode_reward = batch['episode_reward']

            loglikelihood = self.flow_likelihood(action, state)
            prior_prob = self.prior_likelihood(action) if self.use_prior else 0
            assert loglikelihood.shape == episode_reward.shape

            # this term is actually the log-likelihood weighted by reward
            loss_r = -(loglikelihood * (torch.exp(-episode_reward) + self.alpha * prior_prob)).mean()
            self.optimizer.zero_grad()
            loss_r.backward()
            self.optimizer.step()

    def save_model(self):
        if not os.path.exists(self.model_path):
            self.logger.log(f'>> Creating folder for saving model: {self.model_path}')
            os.makedirs(self.model_path)
        model_filename = os.path.join(self.model_path, f'{self.model_id}.pt')
        self.logger.log(f'>> Saving nf model to {model_filename}')
        with open(model_filename, 'wb+') as f:
            torch.save({'parameters': self.model.state_dict()}, f)

    def load_model(self, scenario_configs=None):
        assert scenario_configs is not None, 'Scenario configs should be provided for loading model.'
        scenario_id = scenario_configs[0].scenario_id
        model_file = scenario_configs[0].parameters[0]
        for config in scenario_configs:
            assert scenario_id == config.scenario_id, 'Scenarios should be the same in a batch.'
            assert model_file == config.parameters[0], 'Model filenames should be the same in a batch.'
        model_filename = os.path.join(self.model_path, str(scenario_id), model_file)
        if os.path.exists(model_filename):
            self.logger.log(f'>> Loading nf model from {model_filename}')
            with open(model_filename, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['parameters'])
        else:
            self.logger.log(f'>> Fail to find nf model from {model_filename}', color='yellow')

    def save_prior_model(self):
        states = {'parameters': self.prior_model.state_dict()}
        model_filename = os.path.join(self.model_path, 'nf.prior.'+str(self.model_id)+'.pt')
        with open(model_filename, 'wb+') as f:
            torch.save(states, f)
            self.logger.log(f'>> Save prior model of nf')

    def load_prior_model(self):
        model_filename = os.path.join(self.model_path, 'nf.prior.'+str(self.model_id)+'.pt')
        self.logger.log(f'>> Loading nf model from {model_filename}')
        if os.path.isfile(model_filename):
            with open(model_filename, 'rb') as f:
                checkpoint = torch.load(f)
            self.prior_model.load_state_dict(checkpoint['parameters'])
        else:
            self.logger.log(f'>> Fail to find nf prior model from {model_filename}', color='yellow')
