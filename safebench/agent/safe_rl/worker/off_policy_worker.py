import numpy as np
import torch
from tqdm import tqdm
from cpprb import ReplayBuffer
from safebench.util.torch_util import to_tensor


class OffPolicyWorker:
    """
        Collect data based on the policy and env, and store the interaction data to data buffer.
    """

    def __init__(self, config, logger):
        self.env = None
        self.policy = None
        self.data_loader = None
        self.logger = logger
        self.timeout_steps = config['timeout_steps']
        self.warmup_steps = config['warmup_steps']

        self.episode_rerun_num = config['episode_rerun_num']
        self.sample_episode_num = config['sample_episode_num']
        self.evaluate_episode_num = config['evaluate_episode_num']
        self.batch_size = config['batch_size']
        self.verbose = config['verbose']

        obs_dim = config['ego_state_dim']
        act_dim = config['ego_action_dim']
        self.obs_type = config['obs_type']

        env_dict = {
            'act': {
                'dtype': np.float32,
                'shape': act_dim
            },
            'done': {
                'dtype': np.float32,
            },
            'obs': {
                'dtype': np.float32,
                'shape': obs_dim
            },
            'obs2': {
                'dtype': np.float32,
                'shape': obs_dim
            },
            'rew': {
                'dtype': np.float32,
            }
        }
        self.cpp_buffer = ReplayBuffer(config['buffer_size'], env_dict)

    def set_environment(self, env, agent, data_loader):
        self.env = env
        self.policy = agent
        self.data_loader = data_loader

        ######### Warmup phase to collect data with random policy #########
        steps = 0
        while steps < self.warmup_steps:
            steps += self.work(warmup=True)

        ######### Train the policy with warmup samples #########
        for i in range(self.warmup_steps // 2):
            self.policy.learn_on_batch(self.get_sample())

    def train_one_epoch(self, epoch, total_epochs):
        epoch_steps = 0
        range_instance = tqdm(
            range(self.sample_episode_num // self.data_loader.num_scenario),
            desc='Collecting trajectories'
        ) if self.verbose else range(self.sample_episode_num // self.data_loader.num_scenario)
        for i in range_instance:
            steps = self.work()
            epoch_steps += steps

        train_steps = self.episode_rerun_num * epoch_steps // self.batch_size
        range_instance = tqdm(
            range(train_steps), desc='training {}/{}'.format(
                epoch + 1, total_epochs)) if self.verbose else range(train_steps)
        for i in range_instance:
            data = self.get_sample()
            self.policy.learn_on_batch(data)

        return epoch_steps

    def work(self, warmup=False):
        '''
        Interact with the environment to collect data
        '''
        # sample scenarios
        sampled_scenario_configs, num_sampled_scenario = self.data_loader.sampler()
        # reset envs
        obss = self.env.reset(sampled_scenario_configs)

        for i in range(self.timeout_steps):
            if self.env.all_scenario_done():
                break

            if warmup:
                action = self.env.sample_action_space()
            else:
                action, _ = self.policy.act(obss, deterministic=False, with_logprob=False)
            obss, reward, done, info = self.env.step(action)  # assume action in [-1, 1]
        self.env.clean_up()

        ep_len_total = 0
        for trajectory in self.env.replay_buffer.get_trajectories():
            ep_reward = ep_cost = 0
            for timestep in trajectory:
                obs = timestep['obs']
                action = timestep['act']
                obs_next = timestep['obs2']
                reward = timestep['rew']
                done = timestep['done']
                info = timestep['info']

                self.logger.store(Act1=action[0], Act2=action[1], tab="worker")
                if "cost" in info:
                    cost = info["cost"]
                    ep_cost += cost
                    self.cpp_buffer.add(obs=obs,
                                        act=np.squeeze(action),
                                        rew=reward,
                                        obs2=obs_next,
                                        done=done,
                                        cost=cost)
                else:
                    self.cpp_buffer.add(obs=obs,
                                        act=np.squeeze(action),
                                        rew=reward,
                                        obs2=obs_next,
                                        done=done)
                ep_reward += reward
                ep_len_total += 1
            self.logger.store(EpRet=ep_reward, EpCost=ep_cost, tab="worker")
        return ep_len_total

    def eval(self):
        '''
        Interact with the environment to collect data
        '''
        # sample scenarios
        sampled_scenario_configs, num_sampled_scenario = self.data_loader.sampler()
        # reset envs
        obss = self.env.reset(sampled_scenario_configs)

        for i in range(self.timeout_steps):
            if self.env.all_scenario_done():
                break
            action, _ = self.policy.act(obss, deterministic=True, with_logprob=False)
            obss, reward, done, info = self.env.step(action)  # assume action in [-1, 1]
        self.env.clean_up()

        for trajectory in self.env.replay_buffer.get_trajectories():
            ep_reward = ep_len = ep_cost = 0
            for timestep in trajectory:
                obs = timestep['obs']
                action = timestep['act']
                obs_next = timestep['obs2']
                reward = timestep['rew']
                done = timestep['done']
                info = timestep['info']
                if "cost" in info:
                    cost = info["cost"]
                    ep_cost += cost
                ep_reward += reward
                ep_len += 1
            self.logger.store(TestEpRet=ep_reward, TestEpLen=ep_len, TestEpCost=ep_cost, tab="eval")

    def get_sample(self):
        data = to_tensor(self.cpp_buffer.sample(self.batch_size))
        data["rew"] = torch.squeeze(data["rew"])
        data["done"] = torch.squeeze(data["done"])
        if "cost" in data:
            data["cost"] = torch.squeeze(data["cost"])
        return data

    def clear_buffer(self):
        self.cpp_buffer.clear()
