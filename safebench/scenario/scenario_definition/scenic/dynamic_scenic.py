''' 
Date:  
LastEditTime: 
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import math
import carla
import json

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_helper import get_location_in_distance_from_wp


class DynamicScenic(BasicScenario):

    """
    surrounding agents are controlled by scenic
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        super(DynamicScenic, self).__init__("DynamicScenic", config, world)
        self._wmap = CarlaDataProvider.get_map()
        self.timeout = timeout
        self.terminate = False
        
    def _spawn_blocker(self, transform, orientation_yaw):
        pass

    def initialize_actors(self):
        pass

    def update_behavior(self, scenario_action):
        """
        update behavior via scenic
        """
        try:
            next(self.world.scenic.update_behavior)
        except:
            self.terminate = True
        
    def check_scenic_terminate(self):
        """
        This condition is just for small scenarios
        """
        return self.terminate

    def create_behavior(self, scenario_init_action):
        pass