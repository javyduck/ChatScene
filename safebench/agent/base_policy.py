''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-06 23:37:16
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

class BasePolicy:
    name = 'base'
    type = 'unlearnable'

    def __init__(self, config, logger):
        self.ego_vehicles = None

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles

    def train(self, replay_buffer):
        raise NotImplementedError()

    def set_mode(self, mode):
        raise NotImplementedError()

    def get_action(self, state, infos, deterministic):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()

    def save_model(self, episode):
        raise NotImplementedError()
