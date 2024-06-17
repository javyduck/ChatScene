''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 17:22:23
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_operation import ScenarioOperation


class ManeuverOppositeDirection(BasicScenario):
    """
        Vehicle is passing another vehicle in a rural area, in daylight, under clear weather conditions, 
        at a non-junction and encroaches into another vehicle traveling in the opposite direction.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(ManeuverOppositeDirection, self).__init__("ManeuverOppositeDirection-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._first_actor_transform = None
        self._second_actor_transform = None

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 45
        self.ego_max_driven_distance = 200

    def convert_actions(self, actions):
        base_speed = 5.0
        speed_scale = 5.0
        speed = actions[0] * speed_scale + base_speed
        return speed

    def initialize_actors(self):
        first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_actor_waypoint = second_actor_waypoint.get_left_lane()
        first_actor_transform = carla.Transform(first_actor_waypoint.transform.location, first_actor_waypoint.transform.rotation)
        second_actor_transform = second_actor_waypoint.transform

        self.actor_type_list = ['vehicle.nissan.micra', 'vehicle.nissan.micra']
        self.actor_transform_list = [first_actor_transform, second_actor_transform]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario
        
    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action.'
        self._first_vehicle_location = 30
        self._second_vehicle_location = self._first_vehicle_location + 20

    def update_behavior(self, scenario_action):
        # first actor run in low speed, second actor run in normal speed from oncoming route
        opposite_actor_speed = self.convert_actions(scenario_action) 
        self.scenario_operation.go_straight(opposite_actor_speed, 1)

    def check_stop_condition(self):
        pass
