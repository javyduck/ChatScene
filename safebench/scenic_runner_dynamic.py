''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-07 12:28:17
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import copy
import os
import json
import glob
import random

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
from safebench.scenario.scenario_data_loader import ScenarioDataLoader, ScenicDataLoader
from safebench.scenario.tools.scenario_utils import scenario_parse, dynamic_scenic_parse

from safebench.util.logger import Logger, setup_logger_kwargs
from safebench.util.metric_util import get_route_scores, get_perception_scores
from safebench.util.scenic_utils import ScenicSimulator

class ScenicRunner:
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
            self.save_freq = agent_config['save_freq']
            self.logger.create_eval_dir(load_existing_results=False)
        elif self.mode == 'eval':
            self.save_freq = agent_config['save_freq']
            self.logger.log('>> Evaluation Mode, skip config saving', 'yellow')
            self.logger.create_eval_dir(load_existing_results=False)
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

    def _init_world(self):
        self.logger.log(">> Initializing carla world")
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.scenario_config['tm_port'])
        
    def _init_scenic(self, config):
        self.logger.log(f">> Initializing scenic simulator: {config.scenic_file}")
        self.scenic = ScenicSimulator(config.scenic_file, config.extra_params)
        
    def _init_renderer(self):
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN
        if self.scenario_category in ['planning', 'scenic']: 
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

    def run_scenes(self, scenes):
        self.logger.log(f">> Begin to run the scene...")
        ## currently there is only one scene in this list ##
        for scene in scenes:
            if self.scenic.setSimulation(scene):
                self.scenic.update_behavior = self.scenic.runSimulation()
                next(self.scenic.update_behavior)
        
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
            
            scenes = [config.scene for config in sampled_scenario_configs]
            # begin to run the scene
            self.run_scenes(scenes)
            
            # get static obs and then reset with init action 
            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, additional_dict = self.scenario_policy.get_init_action(static_obs)
            obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
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
        
        self.scenic.destroy()
        
    def eval(self, data_loader, select = False):
        num_finished_scenario = 0
        data_loader.reset_idx_counter()
        # recording the score and the id of corresponding selected scenes
        map_id_score = {}
        behavior_name = data_loader.behavior
        opt_step = data_loader.opt_step
        opt_time = 0
        
        log_name = f'OPT_{behavior_name}'
            
        if select:
            self.scene_map[log_name] = {}
            self.scene_map[log_name][f'opt_time_{opt_time}'] = self.scenic.save_params()
            
        while len(data_loader) > 0:
            # sample scenarios
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            num_finished_scenario += num_sampled_scenario
            assert num_sampled_scenario == 1, 'scenic can only run one scene at one time'
            
            scenes = [config.scene for config in sampled_scenario_configs]
            # begin to run the scene
            self.run_scenes(scenes)
            
            # reset envs with new config, get init action from scenario policy, and run scenario
            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, _ = self.scenario_policy.get_init_action(static_obs, deterministic=True)
            obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)

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
                    self.logger.add_frame(pygame.surfarray.array3d(self.display).transpose(1, 0, 2))
                    
                # accumulate scores of corresponding scenario
                reward_idx = 0
                for s_i in infos:
                    score = rewards[reward_idx] if self.scenario_category in ['planning', 'scenic'] else 1-infos[reward_idx]['iou_loss']
                    score_list[s_i['scenario_id']].append(score)
                    reward_idx += 1

            # clean up all things
            self.logger.log(">> All scenarios are completed. Clearning up all actors")
            self.env.clean_up()

            # save video
            if self.save_video:
                data_ids = [config.data_id for config in sampled_scenario_configs]
                self.logger.save_video(data_ids=data_ids, log_name= log_name)

            # print score for ranking
            self.logger.log(f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Ranking scores for batch scenario:', color='yellow')
            for s_i in score_list.keys():
                self.logger.log('\t Env id ' + str(s_i) + ': ' + str(np.mean(score_list[s_i])), color='yellow')

            # calculate evaluation results
            score_function = get_route_scores if self.scenario_category in ['planning', 'scenic'] else get_perception_scores
            all_running_results = self.logger.add_eval_results(records=self.env.running_results)
            all_scores = score_function(all_running_results)
            self.logger.add_eval_results(scores=all_scores)
            self.logger.print_eval_results()
            if len(self.env.running_results) % self.save_freq == 0:
                self.logger.save_eval_results(log_name)
                
            if infos[0]['collision']:
                self.scenic.record_params()
                
            if select and (num_finished_scenario % opt_step == 0):
                opt_time += 1
                self.scenic.update_params()
                self.scene_map[log_name][f'opt_time_{opt_time}'] = self.scenic.save_params()
                data_loader.train_scene(opt_time)
                
        self.logger.save_eval_results(log_name)
        
        if select:
            self.scene_map[log_name]['select_id'] = self.select_adv_scene(self.logger.eval_records, score_function, data_loader.select_num)
            self.dump_scene_map()
            
        self.logger.clear()
        self.scenic.destroy()
    
    def select_adv_scene(self, results, score_function, select_num):
        # define your own selection mechanism here
        map_id_score_collision = {}
        map_id_score_non_collision = {}
        for i in results.keys():
            score = score_function({i:results[i]})
            if score['collision_rate'] == 1:
                map_id_score_collision[i] = score['final_score']
            else:
                map_id_score_non_collision[i] = score['final_score']

        # Sort the collision scenes by their scores
        collision_scenes_sorted = sorted(map_id_score_collision.items(), key=lambda x: x[1])

        # Get the number of scenes to select from the collision cases
        num_collision_selected = min(select_num, len(collision_scenes_sorted))

        # Select the lowest scored scenes with collision
        selected_scene_id = [scene[0] for scene in collision_scenes_sorted[:num_collision_selected]]

        # If not enough collision scenes, select remaining scenes
        num_non_collision_selected = select_num - num_collision_selected
        if num_non_collision_selected > 0:
            # Sort the non-collision scenes by their scores
            non_collision_scenes_sorted = sorted(map_id_score_non_collision.items(), key=lambda x: x[1])
            # Select the lowest scored scenes from the non-collision cases
            selected_scene_id.extend([scene[0] for scene in non_collision_scenes_sorted[:num_non_collision_selected]])
        return sorted(selected_scene_id)

    def run(self, test_epoch = None):
        # get scenario data of different maps
        config_list = dynamic_scenic_parse(self.scenario_config, self.logger)
        
        
        ### load rl model ##
        if self.mode == 'train_scenario':
            ## we only need the pretrained surrogate model here ##
            pass
        elif self.mode == 'train_agent':
            ## initlize buffer ###
            Buffer = RouteReplayBuffer if self.scenario_category in ['scenic', 'planning'] else PerceptionReplayBuffer
            replay_buffer = Buffer(self.num_scenario, self.mode, self.buffer_capacity)
            
            ### repeat the training, 20 is just a random placeholder
            config_list = config_list * 20
            
            ### check if resume ###
            if self.continue_agent_training:
                self.logger.load_training_results()
                start_episode = self.check_continue_training(self.agent_policy, replay_buffer) + 1
                if start_episode >= self.train_episode:
                    return
            else:
                self.clean_cache(self.agent_policy.model_path)
                start_episode = -1
                
        elif self.mode == 'eval' and test_epoch:
            ### load trained model for evaluation ###
            self.agent_policy.load_model(episode=test_epoch)
            
        last_town = None
        for config in config_list:
            
            ## set log name ##
            log_name = f'OPT_{config.behavior}'
            
            ## check if all done ##
            if self.mode == 'eval':
                if self.logger.check_eval_dir(log_name) == config.select_num:
                    self.logger.log(f">> This scenario and route have been done.")
                    continue
            elif self.mode == 'train_agent': 
                if self.current_episode >= self.train_episode - 1:
                    return
                
                if self.current_episode + config.select_num < start_episode:
                    self.current_episode += config.select_num
                    continue
                
            # initialize scenic
            self._init_scenic(config)
            
            # initialize map and render
            self._init_world()
            self._init_renderer()
            self.world.scenic = self.scenic
                
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
            data_loader = ScenicDataLoader(self.scenic, config, self.num_scenario)
            # run with different modes
            
            if self.mode == 'train_scenario':
                ### select hard scenic scenario config on the surrogate model ###
                self.scene_map = self.load_scene_map()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('eval')
                self.eval(data_loader, select = True)
            elif self.mode == 'train_agent':
                ### train the surrogate model on the selected hard scenrios ###
                self.agent_policy.set_mode('train')
                self.scenario_policy.set_mode('eval')
                self.train(data_loader, start_episode, replay_buffer)
            elif self.mode == 'eval':
                ### evaluate the trained agent on different test models ###
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('eval')
                self.eval(data_loader)
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
    
    def dump_scene_map(self):
        # load previous checkpoint
        scenic_dir = self.scenario_config['scenic_dir']
        f = open(os.path.join(scenic_dir, f"dynamic_scenario.json"), 'w')
        json_dumps_str = json.dumps(self.scene_map, indent=4)
        print(json_dumps_str, file=f)
        f.close()
        
    def load_scene_map(self):
        # load previous checkpoint
        scenic_dir = self.scenario_config['scenic_dir']
        try:
            with open(os.path.join(scenic_dir, f"dynamic_scenario.json"), 'r') as f:
                data = json.loads(f.read())
        except:
            data = {}
        return data

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
        pygame.quit() # close pygame renderer
        if self.env:
            self.env.clean_up()
            
    