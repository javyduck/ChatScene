''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-05 14:55:02
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

class BasePolicy:
    name = 'base'
    type = 'unlearnable'

    """ This is the template for implementing the policy for a scenario. """
    def __init__(self, config, logger):
        self.continue_episode = 0

    def train(self, replay_buffer):
        raise NotImplementedError()

    def set_mode(self, mode):
        raise NotImplementedError()

    def get_action(self, state, infos, deterministic):
        raise NotImplementedError()
    
    def get_init_action(self, scenario_config, deterministic=False):
        raise NotImplementedError()

    def load_model(self, scenario_configs=None):
        raise NotImplementedError()
