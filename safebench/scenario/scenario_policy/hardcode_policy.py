''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-22 17:57:34
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os
import json
from safebench.scenario.scenario_policy.base_policy import BasePolicy


class HardCodePolicy(BasePolicy):
    name = 'hardcode'
    type = 'unlearnable'

    def __init__(self, scenario_config, logger):
        self.logger = logger
        self.num_scenario = scenario_config['num_scenario']
        self.model_path = os.path.join(scenario_config['ROOT_DIR'], scenario_config['model_path'])
        self.parameters = []
        self.mode = 'eval'
        self.scenario_type = scenario_config['scenario_type']

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, state, infos, deterministic=False):
        return [None] * self.num_scenario

    def get_init_action(self, state, deterministic=False):
        return self.parameters, None

    def load_model(self, scenario_configs=None):
        self.parameters = []
        for config in scenario_configs:
            scenario_id = config.scenario_id
            parameters = config.parameters
            if isinstance(parameters, str):
                model_file = config.parameters
                model_filename = os.path.join(self.model_path, str(scenario_id), model_file)
                if os.path.exists(model_filename):
                    self.logger.log(f'>> Loading {self.scenario_type} model from {model_filename}')
                    with open(model_filename, 'r') as f:
                        self.parameters.append(json.load(f))
                else:
                    self.logger.log(f'>> Fail to find {self.scenario_type} model from {model_filename}', color='yellow')
            elif isinstance(parameters, list):
                self.parameters.append(None)
            else:
                self.logger.log(f'>> Fail to find {self.scenario_type} parameters', color='yellow')
