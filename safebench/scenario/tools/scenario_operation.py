''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-04 21:05:02
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/tools>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.util.pid_controller import VehiclePIDController
from safebench.scenario.tools.scenario_utils import calculate_distance_locations


class ScenarioOperation(object):
    """
        This class defines some atomic operation for actors. All actor's behaviors should be combination of these operations
    """

    def __init__(self):
        self.other_actors = []
        self.need_accelerated = False
        self.vehicle_controller = {}

    def initialize_vehicle_actors(self, actor_transform_list, actor_type_list):
        other_actor_list = []
        if len(actor_type_list) != len(actor_transform_list):
            print("Error caused by length match:", len(actor_type_list), len(actor_transform_list))
        else:
            for i in range(len(actor_type_list)):
                actor = CarlaDataProvider.request_new_actor(actor_type_list[i], actor_transform_list[i])
                if actor is not None:
                    actor.set_simulate_physics(enabled=True)
                other_actor_list.append(actor)

        self.other_actors = other_actor_list
        self._init_vehicle_controller()
        return self.other_actors

    def _init_vehicle_controller(self):
        fps = 30
        _dt = 1.0 / fps
        _args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': _dt}
        _args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': _dt}
        for i in range(len(self.other_actors)):
            if(isinstance(self.other_actors[i], carla.Vehicle)):
                cur_id = self.other_actors[i].id
                cur_controller = VehiclePIDController(self.other_actors[i], args_lateral=_args_lateral_dict, args_longitudinal=_args_longitudinal_dict)
                self.vehicle_controller[cur_id] = cur_controller

    def go_straight(self, target_speed, i, throttle_value=1.0, break_value=1.0, steering=0.0):
        control = self.other_actors[i].get_control()
        if CarlaDataProvider.get_velocity(self.other_actors[i]) <= target_speed:
            self.need_accelerated = True
        else:
            self.need_accelerated = False
        if self.need_accelerated:
            control.throttle = throttle_value
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = break_value
        control.steer = steering
        self.other_actors[i].apply_control(control)

    def walker_go_straight(self, target_speed, i):
        control = self.other_actors[i].get_control()
        control.speed = target_speed
        control.direction = CarlaDataProvider.get_transform(self.other_actors[i]).get_forward_vector()
        # control.throttle = 1.0
        self.other_actors[i].apply_control(control)

    def drive_to_target_followlane(self, i ,target_transform, target_speed):
        # 'i' represents id/order of specific actor in other_actors list
        cur_vehicle_control = self.vehicle_controller.get(self.other_actors[i].id)
        control = cur_vehicle_control.run_step(target_speed, target_transform)
        self.other_actors[i].apply_control(control)

    def drive_to_nofollowlane(self, i, location_queue, target_speed):
        cur_vehicle_control = self.vehicle_controller.get(self.other_actors[i].id)
        cur_actor_location = CarlaDataProvider.get_location(self.other_actors[i])
        target_location = None
        if (len(location_queue) > 0):
            target_location = location_queue[0]
        if(target_location):
            if(calculate_distance_locations(target_location, cur_actor_location) < 5):
                location_queue.pop(0)
            else:
                target_location = target_location
        target_waypoint = CarlaDataProvider.get_map().get_waypoint(cur_actor_location)
        control = cur_vehicle_control.run_step(target_speed, target_waypoint)
        self.other_actors[i].apply_control(control)

    def brake(self, actor):
        control = actor.get_control()
        control.throttle = 0.0
        control.brake = 1.0
        actor.apply_control(control)

    def roll_over(self):
        pass
