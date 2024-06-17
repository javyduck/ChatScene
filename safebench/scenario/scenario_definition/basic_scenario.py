''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 20:27:22
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenarios/basic_scenario.py>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class BasicScenario(object):
    """
        Base class for user-defined scenario
    """
    def __init__(self, name, config, world):
        self.world = world
        self.name = name
        self.config = config

        self.ego_vehicles = None
        self.reference_actor = None # the actor used for calculating trigger distance
        self.other_actors = []
        self.other_actor_transform = []
        self.trigger_distance_threshold = None
        self.ego_max_driven_distance = 200

        
        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()
    

    def create_behavior(self, scenario_init_action):
        """
            This method defines the initial behavior of the scenario
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def update_behavior(self, scenario_action):
        """
            This method defines how to update the behavior of the actors in scenario in each step.
        """
        raise NotImplementedError(
                "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def initialize_actors(self):
        """
            This method defines how to initialize the actors in scenario.
        """
        raise NotImplementedError(
                "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def check_stop_condition(self):
        """
            This method defines the stop condition of the scenario.
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def clean_up(self):
        """
            Remove all actors
        """
        for s_i in range(len(self.other_actors)):
            if self.other_actors[s_i].type_id.startswith('vehicle'):
                self.other_actors[s_i].set_autopilot(enabled=False, tm_port=CarlaDataProvider.get_traffic_manager_port())
            if CarlaDataProvider.actor_id_exists(self.other_actors[s_i].id):
                CarlaDataProvider.remove_actor_by_id(self.other_actors[s_i].id)
        self.other_actors = []
