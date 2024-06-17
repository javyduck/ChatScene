''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:50:05
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
        by running a red traffic light (while the ego vehicle has green)
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight-CC", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # parameters = [self.actor_speed, self.trigger_distance_threshold]
        # parameters = [10, 35]
        self.parameters = config.parameters
        self.actor_speed = self.parameters[0]

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")
        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = self.parameters[1]
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def initialize_actors(self):
        first_vehicle_transform = carla.Transform(
            carla.Location(
                self.config.other_actors[0].transform.location.x,
                self.config.other_actors[0].transform.location.y,
                self.config.other_actors[0].transform.location.z
            ),
            self.config.other_actors[0].transform.rotation)
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario
        
        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")
        traffic_light_other.set_state(carla.TrafficLightState.Red)
        traffic_light_other.set_red_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self.actor_speed, i)

    def check_stop_condition(self):
        # stops when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionLeftTurn(BasicScenario):
    """
        Vehicle turning left at signalized junction scenario
        An actor has higher priority, ego needs to yield to oncoming actor
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(SignalizedJunctionLeftTurn, self).__init__("SignalizedJunctionLeftTurn-CC", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # parameters = [self._target_vel, self.trigger_distance_threshold]
        # parameters = [12.0, 45]
        self.parameters = config.parameters
        self._map = CarlaDataProvider.get_map()
        self._target_vel = self.parameters[0]
        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        # traffic_light_other = CarlaDataProvider.get_next_traffic_light(config.other_actors[0], True)
        if self._traffic_light is None:
            raise RuntimeError("No traffic light for the given location found")
        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)
        # other vehicle's traffic light

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = self.parameters[1]
        self.ego_max_driven_distance = 150

    def initialize_actors(self):
        first_vehicle_transform = carla.Transform(
            carla.Location(
                self.config.other_actors[0].transform.location.x,
                self.config.other_actors[0].transform.location.y,
                self.config.other_actors[0].transform.location.z
            ),
            self.config.other_actors[0].transform.rotation)
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            raise RuntimeError("No traffic light for the given location found")
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        # just move forward with a specific speed
        for i in range(len(self.other_actors)):
            self.scenario_operation.go_straight(self._target_vel, i)

    def check_stop_condition(self):
        # stops when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionRightTurn(BasicScenario):
    """
        Vehicle turning right at signalized junction scenario
        An actor has higher priority, ego needs to yield to oncoming actor
    """

    def __init__(self, world, ego_vehicle, config, timeout=80):
        super(SignalizedJunctionRightTurn, self).__init__("SignalizedJunctionRightTurn-CC", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # parameters = [self._target_vel, self.trigger_distance_threshold]
        # parameters = [12, 35]
        self.parameters = config.parameters
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._target_vel = self.parameters[0]
        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            raise RuntimeError("No traffic light for the given location found")
        self._traffic_light.set_state(carla.TrafficLightState.Red)
        self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = self.parameters[1]
        self.trigger = False
        self.ego_max_driven_distance = 150

    def initialize_actors(self):
        first_vehicle_transform = carla.Transform(
            carla.Location(
                self.config.other_actors[0].transform.location.x,
                self.config.other_actors[0].transform.location.y,
                self.config.other_actors[0].transform.location.z
            ),
            self.config.other_actors[0].transform.rotation)
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            raise RuntimeError("No traffic light for the given location found")
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self._target_vel, i)

    def check_stop_condition(self):
        # stops when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class NoSignalJunctionCrossingRoute(BasicScenario):
    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(NoSignalJunctionCrossingRoute, self).__init__("NoSignalJunctionCrossingRoute-CC", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # parameters = [self.actor_speed, self.trigger_distance_threshold]
        # parameters = [10, 35]
        self.parameters = config.parameters
        self.actor_speed = self.parameters[0]

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = self.parameters[1]
        self.trigger = False

        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def initialize_actors(self):
        first_vehicle_transform = carla.Transform(
            carla.Location(
                self.config.other_actors[0].transform.location.x,
                self.config.other_actors[0].transform.location.y,
                self.config.other_actors[0].transform.location.z
            ),
            self.config.other_actors[0].transform.rotation)

        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario
        
    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self.actor_speed, i)

    def check_stop_condition(self):
        # stops when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False