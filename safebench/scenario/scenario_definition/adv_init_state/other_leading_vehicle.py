''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-30 12:19:37
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance

from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class OtherLeadingVehicle(BasicScenario):
    """
        The user-controlled ego vehicle follows a leading car driving down a given road. 
        At some point the leading car has to decelerate. The ego vehicle has to react accordingly by changing lane 
        to avoid a collision and follow the leading car in other lane. The scenario ends either via a timeout, 
        or if the ego vehicle drives some distance. (Traffic Scenario 05)
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(OtherLeadingVehicle, self).__init__("OtherLeadingVehicle-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._first_actor_transform = None
        self._second_actor_transform = None

        self.dece_distance = 5
        self.dece_target_speed = 2  # 3 will be safe
        self.need_decelerate = False

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 35

    def convert_actions(self, actions):
        x_min = 30
        x_max = 40
        x_scale = (x_max-x_min)/2

        y_min = 0
        y_max = 5
        y_scale = (y_max-y_min)/2

        yaw_min = 8
        yaw_max = 16
        yaw_scale = (yaw_max-yaw_min)/2

        d_min = 8
        d_max = 16
        d_scale = (d_max-d_min)/2

        x_mean = (x_max + x_min)/2
        y_mean = (y_max + y_min)/2
        yaw_mean = (yaw_max + yaw_min)/2
        dist_mean = (d_max + d_min)/2

        x = actions[0] * x_scale + x_mean
        y = actions[1] * y_scale + y_mean
        yaw = actions[2] * yaw_scale + yaw_mean
        dist = actions[3] * d_scale + dist_mean
        return [x, y, yaw, dist]

    def initialize_actors(self):
        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_vehicle_waypoint = second_vehicle_waypoint.get_left_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location, first_vehicle_waypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location, second_vehicle_waypoint.transform.rotation)
        
        self.actor_type_list = ['vehicle.nissan.patrol', 'vehicle.audi.tt']
        self.actor_transform_list = [first_vehicle_transform, second_vehicle_transform]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        # At specific point, vehicle in front of ego will decelerate other_actors[0] is the vehicle before the ego
        # cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.ego_vehicles[0]), CarlaDataProvider.get_transform(self.other_actors[1]))
        cur_distance = calculate_distance_transforms(self.actor_transform_list[0], CarlaDataProvider.get_transform(self.other_actors[0]))
        if cur_distance > self.dece_distance:
            self.need_decelerate = True
        for i in range(len(self.other_actors)):
            if i == 0 and self.need_decelerate:
                self.scenario_operation.go_straight(self.dece_target_speed, i)
            else:
                self.scenario_operation.go_straight(self.other_actor_speed[i], i)

    def create_behavior(self, scenario_init_action):
        self.actions = self.convert_actions(scenario_init_action)
        x1, x2, v1, v2 = self.actions  # [35, 1, 12, 12]
        self._first_vehicle_location = x1
        self._second_vehicle_location = self._first_vehicle_location + x2
        # self._ego_vehicle_drive_distance = self._first_vehicle_location * 4
        self._first_vehicle_speed = v1
        self._second_vehicle_speed = v2
        self.other_actor_speed = [v1, v2]

    def check_stop_condition(self):
        pass
