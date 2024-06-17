'''
 
Email: 
Date: 2023-01-30 22:30:20
LastEditTime: 2023-02-26 00:37:49
Description: 
'''
import os
import numpy as np
from fnmatch import fnmatch
from safebench.util.run_util import setup_eval_configs
from safebench.agent.safe_rl.policy import DDPG, PPO, SAC, TD3
from safebench.agent.safe_rl.worker import OffPolicyWorker, OnPolicyWorker


# a list of implemented algorithms
POLICY_LIST = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}


WORKER_LIST = {
    "ppo": OnPolicyWorker,
    "sac": OffPolicyWorker,
    "td3": OffPolicyWorker,
    "ddpg": OffPolicyWorker,
}


class RLAgent:
    """ 
        Works as an wrapper for all RL agents.
    """
    type = 'offpolicy'
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.policy_name = config['policy_name']
        self.ego_action_dim = config['ego_action_dim']
        self.continue_episode = 0
        self.mode = config['mode']
        self.batch_size = config['batch_size']
        self.buffer_start_training = config['buffer_start_training']
        
        policy_config = config[self.policy_name]
        policy_config['ego_state_dim'] = config['ego_state_dim']
        policy_config['ego_action_dim'] = config['ego_action_dim']
        policy_config['ego_action_limit'] = config['ego_action_limit']
        self.policy = POLICY_LIST[self.policy_name](policy_config, logger)
        self.model_path = os.path.join(config['ROOT_DIR'], config['load_dir'])
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        if config['pretrain_dir'] != None:
            self.logger.log(f'>> Initilizing {self.policy_name} model from {config["pretrain_dir"]}')
            self.policy.load_model(config['pretrain_dir'])
#         self.load_itr = self.load_model()

    def set_mode(self, mode):
        self.mode = mode
#         assert mode == 'eval', 'Safe RL agents only support evaluation mode'
        if mode == 'eval':
            self.policy.actor.eval()
            self.policy.critic.eval()
        else:
            self.policy.actor.train()
            self.policy.critic.train()

    def get_action(self, obs, infos, deterministic=True, with_logprob=False):
        action = []
        for i in range(obs.shape[0]):
            res = self.policy.act(obs[i], deterministic=deterministic, with_logprob=with_logprob)
            res[0][1] = - res[0][1]
            action.append(res[0])
        return np.array(action)

#     def load_model(self):
#         if self.mode in ['eval', 'train_scenario']:
#             assert self.config['load_dir'] is not None, "Please specify load_dir!"
#         if self.config['load_dir'] is not None:
#             model_path, load_itr, _, _, _ = setup_eval_configs(self.config['load_dir'], itr=self.config['load_iteration'])
#             self.logger.log(f'>> Loading model from {model_path}')
#             self.policy.load_model(model_path)
#             return load_itr
#         else:
#             return None
        
#     def save_model(self, path = 'safebench/agent/model_ckpt/sac/advsim/model.pt'):
#         self.policy.save_model(model_path)

    def load_model(self, episode=None, replay_buffer=None):
        if episode is None:
            episode = -1
            for _, _, files in os.walk(self.model_path):
                for name in files:
                    if fnmatch(name, "*torch"):
                        cur_episode = int(name.split(".")[-2])
                        if cur_episode > episode:
                            episode = cur_episode
                            
        filepath = os.path.join(self.model_path, f'model.sac.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading {self.policy_name} model from {filepath}')
            self.policy.load_model(filepath)
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No {self.policy_name} model found at {filepath}', 'red')

        if replay_buffer != None and episode != -1:
            buffer_path = os.path.join(self.model_path, f'model.sac.buffer')
            if os.path.isfile(buffer_path):
                self.logger.log(f'>> Loading {self.policy_name} buffer from {buffer_path}')
                replay_buffer.load_buffer(buffer_path)
            else:
                self.logger.log(f'>> No {self.policy_name} buffer found at {buffer_path}', 'red')
            
    def save_model(self, episode, replay_buffer = None):
        filepath = os.path.join(self.model_path, f'model.sac.{episode:04}.torch')
        self.logger.log(f'>> Saving {self.policy_name} model to {filepath}')
        self.policy.save_model(filepath)

        if replay_buffer != None:
            buffer_path = os.path.join(self.model_path, f'model.sac.buffer')
            replay_buffer.save_buffer(buffer_path)
                
    # TODO: expose APIs inside policy
    def train(self, replay_buffer):
        if (replay_buffer.buffer_len-1) + (replay_buffer.collision_buffer_len-1) < self.buffer_start_training:
            return None
#         if replay_buffer.buffer_len < self.buffer_start_training:
#             return None
#         if replay_buffer.collision_buffer_len <= self.buffer_start_training//2:
#             return None
        batch = replay_buffer.sample(self.batch_size)
        loss = self.policy.learn_on_batch(batch)
        return loss
    
    def act(self, obs, deterministic=False, with_logprob=False):
        results = []
        for i in range(obs.shape[0]):
            res = self.policy.act(obs[i], deterministic=deterministic, with_logprob=with_logprob)
            results.append(res)
        return_tuple = []
        for i in range(len(results[0])):
            return_tuple.append(np.array([res[i] for res in results]))
        return return_tuple

    def set_ego_and_route(self, ego_vehicles, info):
        pass
        # self.ego_vehicles = ego_vehicles