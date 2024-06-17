'''
 
Email: 
Date: 2023-02-16 11:20:54
LastEditTime: 2023-02-26 00:38:28
Description: 
'''

import time

from safebench.agent.safe_rl.rl_agent import WORKER_LIST


class AgentTrainer:
    def __init__(self, agent_config, logger):
        self.logger = logger
        self.verbose = agent_config['verbose']
        self.epochs = agent_config['epochs']
        self.evaluate_episode_num = agent_config['evaluate_episode_num']
        self.save_freq = agent_config['save_freq']
        self.policy_config = agent_config[agent_config['policy_name']]
        self.worker_config = self.policy_config["worker_config"]
        self.worker_config['ego_action_dim'] = agent_config['ego_action_dim']
        self.worker_config['ego_state_dim'] = agent_config['ego_state_dim']
        self.worker_config['obs_type'] = agent_config['obs_type']
        self.worker = WORKER_LIST[agent_config['policy_name']](self.worker_config, self.logger)
        self.env = None
        self.agent_policy = None
        self.scenario_policy = None
        self.data_loader = None
        self.replay_buffer = None

    def set_environment(self, env, data_loader, replay_buffer, agent_policy, scenario_policy):
        self.env = env
        self.data_loader = data_loader
        self.replay_buffer = replay_buffer
        self.agent_policy = agent_policy
        self.scenario_policy = scenario_policy
        self.worker.set_environment(env, data_loader, replay_buffer, agent_policy, scenario_policy)

    def train(self):
        continue_from_epoch = self.agent_policy.load_itr
        if continue_from_epoch is None:
            continue_from_epoch = 0
        start_time = time.time()
        total_steps = 0
        for epoch in range(continue_from_epoch, self.epochs):
            epoch_steps = self.worker.train_one_epoch(epoch, self.epochs)
            total_steps += epoch_steps

            for _ in range(self.evaluate_episode_num):
                self.worker.eval()

            if hasattr(self.agent_policy, "post_epoch_process"):
                self.agent_policy.post_epoch_process()

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.logger.save_state({'env': None}, None)
            
            # Log info about epoch
            self.data_dict = self._log_metrics(epoch, total_steps, time.time() - start_time, self.verbose)

    def _log_metrics(self, epoch, total_steps, time=None, verbose=True):
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('TotalStep', total_steps)
        for key in self.logger.logger_keys:
            self.logger.log_tabular(key, average_only=True)
        if time is not None:
            self.logger.log_tabular('Time', time)
        # data_dict contains all the keys except Epoch and TotalEnvInteracts
        data_dict = self.logger.dump_tabular(
            x_axis="TotalStep",
            verbose=verbose,
        )
        return data_dict
