''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:42:53
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients>

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import torch as T
import os
from .networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
            self, 
            actor_dims=7, 
            critic_dims=11, 
            n_actions=2, 
            n_agents=2, 
            agent_idx=1, 
            chkpt_dir='checkpoints/',
            alpha=0.001, 
            beta=0.001, 
            fc1=64, 
            fc2=64, 
            gamma=0.99, 
            tau=0.005
        ):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        
        chkpt_dir = os.path.join(os.path.dirname(__file__), chkpt_dir)
        
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic')
        self.update_network_parameters(tau=1)

    def train(self):
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()
        
    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()
        
    def choose_action(self, observation, action_std = 0.3):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.randn(self.n_actions).to(self.actor.device) * action_std
        action = actions + noise
        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        self.soft_update(self.target_actor, self.actor, tau)
        self.soft_update(self.target_critic, self.critic, tau)
        
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()