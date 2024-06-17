import numpy as np
from safebench.util.torch_util import to_tensor
from safebench.agent.safe_rl.worker.buffer import OnPolicyBuffer


class OnPolicyWorker:
    """
    Collect data based on the policy and env, and store the interaction data to data buffer.
    """
    def __init__(self, config, logger):
        self.env = None
        self.policy = None
        self.data_loader = None
        self.logger = logger
        self.interact_steps = config['interact_steps']
        self.timeout_steps = config['timeout_steps']

        obs_dim = config['ego_state_dim']
        act_dim = config['ego_action_dim']
        self.obs_type = config['obs_type']

        # if "Safe" in env.spec.id:
        #     self.SAFE_RL_ENV = True

        self.buffer = OnPolicyBuffer(obs_dim, act_dim, self.interact_steps + 1, gamma, lam)

    def set_environment(self, env, agent, data_loader):
        self.env = env
        self.policy = agent
        self.data_loader = data_loader

    def train_one_epoch(self, epoch, total_epochs):
        epoch_steps = 0
        steps = self.work()
        epoch_steps += steps
        data = self.get_sample()
        self.policy.learn_on_batch(data)
        return epoch_steps

    def work(self):
        '''
        Interact with the environment to collect data
        '''
        # sample scenarios
        self.cost_list = []
        sampled_scenario_configs, num_sampled_scenario = self.data_loader.sampler()
        # reset envs
        obss = self.env.reset(sampled_scenario_configs)

        for i in range(self.interact_steps):
            if self.env.all_scenario_done():
                break

            action, value, log_prob = self.policy.act(obss)
            obss, reward, done, info = self.env.step(action, value, log_prob)  # assume action in [-1, 1]
            cost_value = 0
            cost = info["cost"] if "cost" in info else 0
        self.env.clean_up()

        for trajectory in self.env.replay_buffer.get_trajectories():
            ep_reward = ep_len = ep_cost = 0
            for i, timestep in enumerate(trajectory):
                obs = timestep['obs']
                action = timestep['act']
                obs_next = timestep['obs2']
                reward = timestep['rew']
                done = timestep['done']
                info = timestep['info']
                value = timestep['value']
                log_prob = timestep['log_prob']
                cost_value = 0

                if done and 'TimeLimit.truncated' in info:
                    done = False
                    timeout_env = True
                else:
                    timeout_env = False

                cost = info["cost"] if "cost" in info else 0

                self.buffer.store(obs, np.squeeze(action), reward, value, log_prob, done, cost, cost_value)
                self.logger.store(VVals=value, CostVVals=cost_value, tab="worker")
                ep_reward += reward
                ep_cost += cost
                ep_len += 1
                obs = obs_next

                timeout = ep_len == self.timeout_steps - 1 or i == self.interact_steps - 1 or timeout_env and not done
                terminal = done or timeout
                if terminal:
                    # after each episode
                    if timeout:
                        # if trajectory didn't reach terminal state, bootstrap value target
                        _, value, _ = self.policy.act(np.array([obs]))
                        value = value[0]
                        cost_value = 0
                    else:
                        value = 0
                        cost_value = 0
                    self.buffer.finish_path(value, cost_value)
                    if i < self.interact_steps - 1:
                        self.logger.store(EpRet=ep_reward, EpLen=ep_len, EpCost=ep_cost, tab="worker")
                    self.cost_list.append(ep_cost)

        return self.interact_steps

    def eval(self):
        '''
        Evaluate the policy
        '''
        sampled_scenario_configs, num_sampled_scenario = self.data_loader.sampler()
        # reset envs
        obss = self.env.reset(sampled_scenario_configs)

        for i in range(self.timeout_steps):
            if self.env.all_scenario_done():
                break

            action, _, _ = self.policy.act(obss, deterministic=True)
            obss, reward, done, info = self.env.step(action)  # assume action in [-1, 1]
        self.env.clean_up()

        for trajectory in self.env.replay_buffer.get_trajectories():
            ep_reward = ep_len = ep_cost = 0
            for i, timestep in enumerate(trajectory):
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
        data = self.buffer.get()
        # torch.save(data, "buffer.pt")
        self.buffer.clear()
        data["ep_cost"] = to_tensor(np.mean(self.cost_list))
        return data
