''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 22:35:17
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''
import os
import copy
import glob

import numpy as np
import carla
import pygame
from tqdm import tqdm

from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.gym_carla.replay_buffer import RouteReplayBuffer, PerceptionReplayBuffer

from safebench.agent import AGENT_POLICY_LIST
from safebench.scenario import SCENARIO_POLICY_LIST

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_data_loader import ScenarioDataLoader
from safebench.scenario.tools.scenario_utils import scenario_parse

from safebench.util.logger import Logger, setup_logger_kwargs
from safebench.util.metric_util import get_route_scores, get_perception_scores


class CarlaRunner:
    def __init__(self, agent_config, scenario_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config

        self.seed = scenario_config['seed']
        self.exp_name = scenario_config['exp_name']
        self.output_dir = scenario_config['output_dir']
        self.mode = scenario_config['mode']
        self.save_video = scenario_config['save_video']

        self.render = scenario_config['render']
        self.num_scenario = scenario_config['num_scenario']
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']
        self.scenario_category = scenario_config['scenario_category']

        # continue training flag
        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        self.client.set_timeout(10.0)
        self.world = None
        self.env = None

        self.env_params = {
            'auto_ego': scenario_config['auto_ego'],
            'obs_type': agent_config['obs_type'],
            'scenario_category': self.scenario_category,
            'ROOT_DIR': scenario_config['ROOT_DIR'],
            'warm_up_steps': 9,                                        # number of ticks after spawning the vehicles
            'disable_lidar': True,                                     # show bird-eye view lidar or not
            'display_size': 128,                                       # screen size of one bird-eye view window
            'obs_range': 32,                                           # observation range (meter)
            'd_behind': 12,                                            # distance behind the ego vehicle (meter)
            'max_past_step': 1,                                        # the number of past steps to draw
            'discrete': False,                                         # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],                          # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],                        # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],                     # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],                     # continuous steering angle range
            'max_episode_step': scenario_config['max_episode_step'],   # maximum timesteps per episode
            'max_waypt': 12,                                           # maximum number of waypoints
            'lidar_bin': 0.125,                                        # bin size of lidar sensor (meter)
            'out_lane_thres': 4,                                       # threshold for out of lane (meter)
            'desired_speed': 8,                                        # desired speed (m/s)
            'image_sz': 1024,                                          # TODO: move to config of od scenario
        }

        # pass config from scenario to agent
        agent_config['mode'] = scenario_config['mode']
        agent_config['ego_action_dim'] = scenario_config['ego_action_dim']
        agent_config['ego_state_dim'] = scenario_config['ego_state_dim']
        agent_config['ego_action_limit'] = scenario_config['ego_action_limit']

        # define logger
        logger_kwargs = setup_logger_kwargs(
            self.exp_name, 
            self.output_dir, 
            self.seed,
            agent=agent_config['policy_type'],
            scenario=scenario_config['policy_type'],
            scenario_category=self.scenario_category
        )
        self.logger = Logger(**logger_kwargs)
        
        # prepare parameters
        if self.mode == 'train_agent':
            self.buffer_capacity = agent_config['buffer_capacity']
            self.eval_in_train_freq = agent_config['eval_in_train_freq']
            self.save_freq = agent_config['save_freq']
            self.train_episode = agent_config['train_episode']
            self.current_episode = -1
            self.logger.save_config(agent_config)
            self.logger.create_training_dir()
        elif self.mode == 'train_scenario':
            self.buffer_capacity = scenario_config['buffer_capacity']
            self.eval_in_train_freq = scenario_config['eval_in_train_freq']
            self.save_freq = scenario_config['save_freq']
            self.train_episode = scenario_config['train_episode']
            self.logger.save_config(scenario_config)
            self.logger.create_training_dir()
        elif self.mode == 'eval':
            self.save_freq = scenario_config['save_freq']
            self.logger.log('>> Evaluation Mode, skip config saving', 'yellow')
            self.logger.create_eval_dir(load_existing_results=True)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

        # define agent and scenario
        self.logger.log('>> Agent Policy: ' + agent_config['policy_type'])
        self.logger.log('>> Scenario Policy: ' + scenario_config['policy_type'])

        if self.scenario_config['auto_ego']:
            self.logger.log('>> Using auto-polit for ego vehicle, action of policy will be ignored', 'yellow')
        if scenario_config['policy_type'] == 'ordinary' and self.mode != 'train_agent':
            self.logger.log('>> Ordinary scenario can only be used in agent training', 'red')
            raise Exception()
        self.logger.log('>> ' + '-' * 40)

        # define agent and scenario policy
        self.agent_policy = AGENT_POLICY_LIST[agent_config['policy_type']](agent_config, logger=self.logger)
        self.scenario_policy = SCENARIO_POLICY_LIST[scenario_config['policy_type']](scenario_config, logger=self.logger)
        if self.save_video:
            assert self.mode == 'eval', "only allow video saving in eval mode"
            self.logger.init_video_recorder()

    def _init_world(self, town):
        self.logger.log(f">> Initializing carla world: {town}")
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.scenario_config['tm_port'])
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _init_renderer(self):
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN
        if self.scenario_category == 'planning': 
            # [bird-eye view, Lidar, front view] or [bird-eye view, front view]
            if self.env_params['disable_lidar']:
                window_size = (self.env_params['display_size'] * 2, self.env_params['display_size'] * self.num_scenario)
            else:
                window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * self.num_scenario)
        else:
            window_size = (self.env_params['display_size'], self.env_params['display_size'] * self.num_scenario)
        self.display = pygame.display.set_mode(window_size, flag)

        # initialize the render for generating observation and visualization
        pixels_per_meter = self.env_params['display_size'] / self.env_params['obs_range']
        pixels_ahead_vehicle = (self.env_params['obs_range'] / 2 - self.env_params['d_behind']) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.env_params['display_size'], self.env_params['display_size']],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params, logger=self.logger)

    def train(self, data_loader, start_episode=0, replay_buffer = None):
        # general buffer for both agent and scenario

        for _ in tqdm(range(len(data_loader))):
            self.current_episode += 1
            if self.current_episode >= self.train_episode:
                return
            if self.current_episode < start_episode:
                continue
            # sample scenarios
            sampled_scenario_configs, _ = data_loader.sampler()
            # reset the index counter to create endless loader
            # data_loader.reset_idx_counter()

            # get static obs and then reset with init action 
            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, additional_dict = self.scenario_policy.get_init_action(static_obs)
            try:
                obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
            except:
                continue
            replay_buffer.store_init([static_obs, scenario_init_action], additional_dict=additional_dict)

            # get ego vehicle from scenario
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            # start loop
            episode_reward = []
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)

                # apply action to env and get obs
                next_obs, rewards, dones, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)
                replay_buffer.store([ego_actions, scenario_actions, obs, next_obs, rewards, dones], additional_dict=infos)
                obs = copy.deepcopy(next_obs)
                episode_reward.append(np.mean(rewards))

            # train off-policy agent or scenario
            if self.mode == 'train_agent' and self.agent_policy.type == 'offpolicy':
                loss = self.agent_policy.train(replay_buffer)
            elif self.mode == 'train_scenario' and self.scenario_policy.type == 'offpolicy':
                self.scenario_policy.train(replay_buffer)
                
            score_function = get_route_scores if self.scenario_category in ['planning', 'scenic'] else get_perception_scores
            all_scores = score_function(self.env.running_results)
            
            # end up environment
            self.env.clean_up()
            replay_buffer.finish_one_episode()
            self.logger.add_training_results('episode', self.current_episode)
            self.logger.add_training_results('episode_reward', np.sum(episode_reward))
            for key, value in all_scores.items():
                self.logger.add_training_results(key, value)

            if loss is not None:
                critic_loss, actor_loss = loss
                self.logger.add_training_results('critic_loss', critic_loss)
                self.logger.add_training_results('actor_loss', actor_loss)
            else:
                critic_loss, actor_loss = 0, 0
                self.logger.add_training_results('critic_loss', critic_loss)
                self.logger.add_training_results('actor_loss', actor_loss)  
            self.logger.log(f">> Episode: {self.current_episode}, #buffer_len: {replay_buffer.buffer_len}, critic: {critic_loss:.3f}, actor: {actor_loss:.3f}")
            self.logger.save_training_results()

            # train on-policy agent or scenario
            if self.mode == 'train_agent' and self.agent_policy.type == 'onpolicy':
                self.agent_policy.train(replay_buffer)
            elif self.mode == 'train_scenario' and self.scenario_policy.type in ['init_state', 'onpolicy']:
                self.scenario_policy.train(replay_buffer)

            # eval during training
            if (self.current_episode+1) % self.eval_in_train_freq == 0:
                #self.eval(env, data_loader)
                pass

            # save checkpoints
            if (self.current_episode+1) % self.save_freq == 0:
                if self.mode == 'train_agent':
                    self.agent_policy.save_model(self.current_episode, replay_buffer)
                if self.mode == 'train_scenario':
                    self.scenario_policy.save_model(self.current_episode)
            
    def eval(self, data_loader):
        num_finished_scenario = 0
        data_loader.reset_idx_counter()
        while len(data_loader) > 0:
            # sample scenarios
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            log_name = f'ROUTE-{sampled_scenario_configs[0].route_id-4}'
            num_finished_scenario += num_sampled_scenario
            
            # reset envs with new config, get init action from scenario policy, and run scenario
            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, _ = self.scenario_policy.get_init_action(static_obs, deterministic=True)
            try:
                obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
            except:
                continue
            # get ego vehicle from scenario
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            score_list = {s_i: [] for s_i in range(num_sampled_scenario)}
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=True)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=True)

                # apply action to env and get obs
                obs, rewards, _, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)

                # save video
                if self.save_video:
                    if self.scenario_category == 'planning':
                        self.logger.add_frame(pygame.surfarray.array3d(self.display).transpose(1, 0, 2))
                    else:
                        self.logger.add_frame({s_i['scenario_id']: ego_actions[n_i]['annotated_image'] for n_i, s_i in enumerate(infos)})

                # accumulate scores of corresponding scenario
                reward_idx = 0
                for s_i in infos:
                    score = rewards[reward_idx] if self.scenario_category == 'planning' else 1-infos[reward_idx]['iou_loss']
                    score_list[s_i['scenario_id']].append(score)
                    reward_idx += 1

            # clean up all things
            self.logger.log(">> All scenarios are completed. Clearning up all actors")
            self.env.clean_up()

            # save video
            if self.save_video:
                data_ids = [config.data_id for config in sampled_scenario_configs]
                self.logger.save_video(data_ids=data_ids)

            # print score for ranking
            self.logger.log(f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Ranking scores for batch scenario:', 'yellow')
            for s_i in score_list.keys():
                self.logger.log('\t Env id ' + str(s_i) + ': ' + str(np.mean(score_list[s_i])), 'yellow')

                
            # calculate evaluation results
            score_function = get_route_scores if self.scenario_category == 'planning' else get_perception_scores
            all_running_results = self.logger.add_eval_results(records=self.env.running_results)
            all_scores = score_function(all_running_results)
            self.logger.add_eval_results(scores=all_scores)
            self.logger.print_eval_results()
            if len(self.env.running_results) % self.save_freq == 0:
                self.logger.save_eval_results(log_name)
        self.logger.save_eval_results(log_name)

    def run(self, test_epoch = None):
        # get scenario data of different maps
        config_by_map = scenario_parse(self.scenario_config, self.logger)
        
        map_keys = list(config_by_map.keys())
        if self.mode == 'train_agent':
            Buffer = RouteReplayBuffer if self.scenario_category == 'planning' else PerceptionReplayBuffer
            replay_buffer = Buffer(self.num_scenario, self.mode, self.buffer_capacity)
            map_keys = map_keys * 20
            if self.continue_agent_training:
                self.logger.load_training_results()
                start_episode = self.check_continue_training(self.agent_policy, replay_buffer) + 1
                if start_episode >= self.train_episode:
                    return
            else:
                self.clean_cache(self.agent_policy.model_path)
                start_episode = -1
        else:
            self.agent_policy.load_model(episode=test_epoch)
            
        for m_i in map_keys:
            if self.mode == 'eval':
                log_name = f'ROUTE-{config_by_map[m_i][0].route_id-4}'
                if self.logger.check_eval_dir(log_name) == len(config_by_map[m_i]):
                    self.logger.log(f">> This scenario and route have been done.")
                    continue
            elif self.mode == 'train_agent': 
                if self.current_episode >= self.train_episode - 1:
                    return
                
                if self.current_episode + len(config_by_map[m_i]) < start_episode:
                    self.current_episode += len(config_by_map[m_i])
                    continue
                
            # initialize map and render
            try:
                self._init_world(m_i)
                self._init_renderer()
            except:
                continue
                
            # create scenarios within the vectorized wrapper
            self.env = VectorWrapper(
                self.env_params, 
                self.scenario_config, 
                self.world, 
                self.birdeye_render, 
                self.display, 
                self.logger
            )

            # prepare data loader and buffer
            data_loader = ScenarioDataLoader(config_by_map[m_i], self.num_scenario, m_i, self.world)
            
            # run with different modes
            
            if self.mode == 'eval':
#                 self.agent_policy.load_model(episode=test_epoch)
                # self.scenario_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('eval')
                self.eval(data_loader)
            elif self.mode == 'train_agent':
#                 start_episode = self.check_continue_training(self.agent_policy)
#                 self.scenario_policy.load_model()
                self.agent_policy.set_mode('train')
                self.scenario_policy.set_mode('eval')
                self.train(data_loader, start_episode, replay_buffer)
            elif self.mode == 'train_scenario':
                start_episode = self.check_continue_training(self.scenario_policy)
                self.agent_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('train')
                self.train(data_loader, start_episode)
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    def check_continue_training(self, policy, replay_buffer):
        # load previous checkpoint
        policy.load_model(replay_buffer = replay_buffer)
        if policy.continue_episode == 0:
            start_episode = 0
            self.logger.log('>> Previous checkpoint not found. Training from scratch.')
        else:
            start_episode = policy.continue_episode
            self.logger.log(f'>> Continue training from previous checkpoint, epoch: {start_episode}.')
        return start_episode
    
    def clean_cache(self, path):
        # Get a list of all files in directory
        all_files = glob.glob(os.path.join(path, '*'))

        # Specify the file to keep
        file_to_keep = os.path.join(path, 'model.sac.-001.torch')

        # Remove all files except the one to keep
        for file in all_files:
            if file != file_to_keep:
                os.remove(file)

    def close(self):
        pygame.quit() 
        if self.env:
            self.env.clean_up()