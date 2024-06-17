''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:46:15
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients>

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import numpy as np


class AdvMultiAgentReplayBuffer:
    def __init__(self, max_size = 100000, critic_dims = 11, actor_dims = [4,7], 
        n_actions = [2,2], n_agents = 2, batch_size = 1024):
        self.n_agents = n_agents
        self.collision_buffer = MultiAgentReplayBuffer(max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size//2)
        self.non_collision_buffer = MultiAgentReplayBuffer(max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size//2)
    
    def store_transition(self, collision, raw_obs, state, action, next_action, reward, raw_obs_, state_, done):
        if collision:
            self.collision_buffer.store_transition(raw_obs, state, action, next_action, reward, raw_obs_, state_, done)
        else:
            self.non_collision_buffer.store_transition(raw_obs, state, action, next_action, reward, raw_obs_, state_, done)
        
    def sample_buffer(self):
        actor_states, states, actions, next_actions, rewards, actor_new_states, states_, terminal = self.collision_buffer.sample_buffer()
        actor_states2, states2, actions2, next_actions2, rewards2, actor_new_states2, states_2, terminal2 = self.non_collision_buffer.sample_buffer()
        
        states = np.concatenate((states, states2), axis = 0)
        rewards = np.concatenate((rewards, rewards2), axis = 0)
        states_ = np.concatenate((states_, states_2), axis = 0)
        terminal = np.concatenate((terminal, terminal2), axis = 0)
    
        for agent_idx in range(self.n_agents):
            actor_states[agent_idx] = np.concatenate((actor_states[agent_idx], actor_states2[agent_idx]), axis = 0)
            actor_new_states[agent_idx] = np.concatenate((actor_new_states[agent_idx], actor_new_states2[agent_idx]), axis = 0)
            actions[agent_idx] = np.concatenate((actions[agent_idx], actions2[agent_idx]), axis = 0)
            next_actions[agent_idx] = np.concatenate((next_actions[agent_idx], next_actions2[agent_idx]), axis = 0)
        return actor_states, states, actions, next_actions, rewards, actor_new_states, states_, terminal
    
    def ready(self):
        if self.collision_buffer.ready() and self.non_collision_buffer.ready():
            return True


class MultiAgentReplayBuffer:
    def __init__(self, max_size = 100000, critic_dims = 11, actor_dims=[4,7], 
            n_actions = [2,2], n_agents = 2, batch_size = 512, chkpt_dir=DIR):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.chkpt_dir = chkpt_dir
        
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        self.actor_next_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append( 
                            np.zeros((self.mem_size, self.n_actions[i])))
            self.actor_next_action_memory.append( 
                        np.zeros((self.mem_size, self.n_actions[i])))

    def store_transition(self, raw_obs, state, action, next_action, reward, raw_obs_, state_, done):
        # this introduces a bug: if we fill up the memory capacity and then
        # zero out our actor memory, the critic will still have memories to access
        # while the actor will have nothing but zeros to sample. Obviously
        # not what we intend.
        # In reality, there's no problem with just using the same index
        # for both the actor and critic states. I'm not sure why I thought
        # this was necessary in the first place. Sorry for the confusion!

        #if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()
        
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]
            self.actor_next_action_memory[agent_idx][index] = next_action[agent_idx]
            
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        next_actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])
            next_actions.append(self.actor_next_action_memory[agent_idx][batch])
        return actor_states, states, actions, next_actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        
        # TODO save the buffer into the checkpoints#
#     def save(self):
#         """
#         Save the replay buffer
#         """
#         folder_path = self.chkpt_dir
        
#         np.save(folder_path + '/states.npy', self.states)
#         np.save(folder_path + '/rewards.npy', self.rewards)
#         np.save(folder_path + '/next_states.npy', self.next_states)
#         np.save(folder_path + '/dones.npy', self.dones)
        
#         for index in range(self.n_agents):
#             np.save(folder_path + '/states_actor_{}.npy'.format(index), self.list_actors_states[index])
#             np.save(folder_path + '/next_states_actor_{}.npy'.format(index), self.list_actors_next_states[index])
#             np.save(folder_path + '/actions_actor_{}.npy'.format(index), self.list_actors_actions[index])
            
#     def load(self, folder_path):
#         self.states = np.load(folder_path + '/states.npy')
#         self.rewards = np.load(folder_path + '/rewards.npy')
#         self.next_states = np.load(folder_path + '/next_states.npy')
#         self.dones = np.load(folder_path + '/dones.npy')
        
#         self.list_actors_states = [np.load(folder_path + '/states_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
#         self.list_actors_next_states = [np.load(folder_path + '/next_states_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
#         self.list_actors_actions = [np.load(folder_path + '/actions_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
        
#         self.buffer_counter = dict_info["buffer_counter"]
#         self.n_games = dict_info["n_games"]