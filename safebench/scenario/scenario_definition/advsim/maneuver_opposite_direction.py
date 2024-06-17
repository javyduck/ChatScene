''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:51:34
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla
import json

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_operation import ScenarioOperation


class ManeuverOppositeDirection(BasicScenario):
    """
    "Vehicle Maneuvering In Opposite Direction" (Traffic Scenario 06)
    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        """
        Setup all relevant parameters and create scenario
        obstacle_type -> flag to select type of leading obstacle. Values: vehicle, barrier
        """
        super(ManeuverOppositeDirection, self).__init__("ManeuverOppositeDirection-AdvSim", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        self._first_vehicle_location = 50
        self._second_vehicle_location = self._first_vehicle_location + 30
        self._opposite_speed = 8   # m/s
        self._first_actor_transform = None
        self._second_actor_transform = None

        self.scenario_operation = ScenarioOperation()

        self.actor_type_list = ['vehicle.nissan.micra', 'vehicle.nissan.micra']

        self.reference_actor = None
        self.trigger_distance_threshold = 45
        self.ego_max_driven_distance = 200

        self.step = 0
        self.control_seq = []
        self._other_actor_max_velocity = self._opposite_speed * 2


    def initialize_actors(self):
        """
        Custom initialization
        """
        first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_actor_waypoint = second_actor_waypoint.get_left_lane()

        first_actor_transform = carla.Transform(
            first_actor_waypoint.transform.location,
            first_actor_waypoint.transform.rotation)

        self.actor_transform_list = [first_actor_transform, second_actor_waypoint.transform]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)

        self.reference_actor = self.other_actors[0]

    def update_behavior(self, scenario_action):
        """
        first actor run in low speed
        second actor run in normal speed from oncoming route
        """
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'
        current_velocity = self.control_seq[self.step if self.step < len(self.control_seq) else -1] * self._other_actor_max_velocity
        self.step += 1
        self.scenario_operation.go_straight(current_velocity, 1)

    def create_behavior(self, scenario_init_action):
        self.control_seq = scenario_init_action

    def check_stop_condition(self):
        pass
