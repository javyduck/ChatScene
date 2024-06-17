''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-30 12:19:04
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms


class OppositeVehicleRunningRedLight(BasicScenario):
    """
        This class holds everything required for a scenario, in which an other vehicle takes priority from the ego vehicle, 
        by running a red traffic light (while the ego vehicle has green).
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location of the ego vehicle found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        """ Process the action from model. action is assumed in [-1, 1] """
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = actions[0] * y_scale + y_mean
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    def initialize_actors(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z),
            other_actor_transform.rotation
        )

        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location of the other vehicle found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Red)
            traffic_light_other.set_red_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        actions = self.convert_actions(scenario_init_action)
        self.x, delta_v, delta_dist = actions  
        self.actor_speed = 10 + delta_v
        self.trigger_distance_threshold = 35 + delta_dist

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self.actor_speed, i)

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionLeftTurn(BasicScenario):
    """
        Vehicle turning left at signalized junction scenario. 
        An actor has higher priority, ego needs to yield to oncoming actor.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(SignalizedJunctionLeftTurn, self).__init__("SignalizedJunctionLeftTurn-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)

        # other vehicle's traffic light
        self.scenario_operation = ScenarioOperation()
        self.reference_actor = None
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = actions[0] * y_scale + y_mean
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    def initialize_actors(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z),
            other_actor_transform.rotation
        )
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Green)
            traffic_light_other.set_green_time(self.timeout)

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        for i in range(len(self.other_actors)):
            self.scenario_operation.go_straight(self._target_vel, i)

    def create_behavior(self, scenario_init_action):
        actions = self.convert_actions(scenario_init_action)
        self.x, delta_v, delta_dist = actions  
        self._target_vel = 12.0 + delta_v
        self.trigger_distance_threshold = 45 + delta_dist

    def check_stop_condition(self):
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionRightTurn(BasicScenario):
    """
        Vehicle turning right at signalized junction scenario an actor has higher priority, ego needs to yield to oncoming actor
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(SignalizedJunctionRightTurn, self).__init__("SignalizedJunctionRightTurn-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Red)
            self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger = False
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = actions[0] * y_scale + y_mean
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    def initialize_actors(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z),
            other_actor_transform.rotation
        )
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Green)
            traffic_light_other.set_green_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        actions = self.convert_actions(scenario_init_action)
        self.x, delta_v, delta_dist = actions  
        self._target_vel = 12 + delta_v
        self.trigger_distance_threshold = 35 + delta_dist

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self._target_vel, i)

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class NoSignalJunctionCrossingRoute(BasicScenario):
    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(NoSignalJunctionCrossingRoute, self).__init__("NoSignalJunctionCrossingRoute-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self.scenario_operation = ScenarioOperation()
        self.reference_actor = None
        
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = actions[0] * y_scale + y_mean
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    def initialize_actors(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z),
            other_actor_transform.rotation
        )
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario
        
    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self.actor_speed, i)

    def create_behavior(self, scenario_init_action):
        actions = self.convert_actions(scenario_init_action)
        self.x, delta_v, delta_dist = actions  
        self.actor_speed = 10 + delta_v
        self.trigger_distance_threshold = 35 + delta_dist

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False
