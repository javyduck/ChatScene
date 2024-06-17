''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-04 01:01:01
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients>

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import torch as T
import os
from .agent import Agent


class MADDPG:
    def __init__(
            self, 
            actor_dims = [4, 7], 
            critic_dims = 11, 
            n_agents = 2, 
            n_actions = [2,2], 
            scenario='standard_scenario3',
            alpha=0.001,
            beta=0.001, 
            fc1=64, 
            fc2=64, 
            gamma=0.99, 
            tau=0.005, 
            chkpt_dir='./checkpoints/'
        ):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario 
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        ### TODO: the first agent should be the ego agent ###
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions[agent_idx], n_agents, agent_idx, alpha=alpha, beta=beta, chkpt_dir=chkpt_dir))
        self.save_checkpoint()
        
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return
        self.train()
        actor_states, states, actions, next_actions, rewards, actor_new_states, states_, dones = memory.sample_buffer()
        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            
            if agent_idx == 0:
                new_pi = T.tensor(next_actions[agent_idx], dtype=T.float).to(device)
            else:
                new_pi = agent.target_actor.forward(new_states)
                
            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            
            if agent_idx == 0:
                pi = T.tensor(actions[agent_idx], dtype=T.float).to(device)
            else:
                pi = agent.actor.forward(mu_states)
                
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(T.tensor(actions[agent_idx], dtype=T.float).to(device))

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            if agent_idx == 0:
                continue
            critic_value_, q_list_ = agent.target_critic.predict(states_, new_actions)
            critic_value_[dones[:,0]] = 0.0
            critic_value, q_list = agent.critic.predict(states, old_actions)

            target = rewards[:,agent_idx] + agent.gamma*critic_value_.flatten()
            critic_loss = sum([((q.flatten() - target)**2).mean() for q in q_list])
    
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss, _ = agent.critic.predict(states, mu)
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
        self.save_checkpoint()
        
    def train(self):
        for agent_idx, agent in enumerate(self.agents):
            agent.train()
        
    def eval(self):
        for agent_idx, agent in enumerate(self.agents):
            agent.eval()
