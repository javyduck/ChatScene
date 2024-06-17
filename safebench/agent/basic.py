''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 19:01:07
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import numpy as np

from safebench.agent.base_policy import BasePolicy
from agents.navigation.basic_agent import BasicAgent  


class CarlaBasicAgent(BasePolicy):
    name = 'basic'
    type = 'unlearnable'

    """ This is just an example for testing, whcih always goes straight. """
    def __init__(self, config, logger):
        self.logger = logger
        self.num_scenario = config['num_scenario']
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'
        self.continue_episode = 0
        self.route = None
        self.controller_list = []

        # parameters for PID controller
        self.target_speed = config['target_speed']
        dt = config['dt']
        lateral_KP = config['lateral_KP']
        lateral_KI = config['lateral_KI']
        lateral_KD = config['lateral_KD']
        longitudinal_KP = config['longitudinal_KP']
        longitudinal_KI = config['longitudinal_KI']
        longitudinal_KD = config['longitudinal_KD']
        max_steering = config['max_steering']
        max_throttle = config['max_throttle']

        self.opt_dict = {
            'lateral_control_dict': {'K_P': lateral_KP, 'K_I': lateral_KI, 'K_D': lateral_KD, 'dt': dt},
            'longitudinal_control_dict': {'K_P': longitudinal_KP, 'K_I': longitudinal_KI, 'K_D': longitudinal_KD, 'dt': dt},
            'max_steering': max_steering,
            'max_throttle': max_throttle,
        }

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles
        self.controller_list = []
        for e_i in range(len(ego_vehicles)):
            controller = BasicAgent(self.ego_vehicles[e_i], target_speed=self.target_speed, opt_dict=self.opt_dict)
            dest_waypoint = info[e_i]['route_waypoints'][-1]
            location = dest_waypoint.transform.location
            controller.set_destination(location) # set route for each controller
            self.controller_list.append(controller)

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for e_i in infos:
            # select the controller that matches the scenario_id
            control = self.controller_list[e_i['scenario_id']].run_step()
            throttle = control.throttle
            steer = control.steer
            actions.append([throttle, steer]) 
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self):
        pass

    def save_model(self):
        pass
