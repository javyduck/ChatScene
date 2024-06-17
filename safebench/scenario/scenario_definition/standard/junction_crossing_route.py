''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:48:34
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

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario


class OppositeVehicleRunningRedLight(BasicScenario):
    """
        An other vehicle takes priority from the ego vehicle, by running a red traffic light (while the ego vehicle has green).
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight", config, world)
        self.timeout = timeout
        self.ego_vehicle = ego_vehicle

        self.actor_speed = 10
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)

        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 35
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def initialize_actors(self):
        config = self.config
        first_vehicle_transform = carla.Transform(
            carla.Location(
                config.other_actors[0].transform.location.x,
                config.other_actors[0].transform.location.y,
                config.other_actors[0].transform.location.z
            ),
            config.other_actors[0].transform.rotation)

        self.actor_type_list = ["vehicle.audi.tt"]
        self.actor_transform_list = [first_vehicle_transform]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0]
        
        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)

        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Red)
            traffic_light_other.set_red_time(self.timeout)

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
        Vehicle turning left at signalized junction scenario. An actor has higher priority, ego needs to yield to oncoming actor.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(SignalizedJunctionLeftTurn, self).__init__("TurnLeftAtSignalizedJunction", config, world)
        self.ego_vehicle = ego_vehicle
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._target_vel = 12.0
        self._actor_distance = 100
        self._traffic_light = None
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            raise RuntimeError("No traffic light for the given location found")
        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 45
        self.ego_max_driven_distance = 150

    def initialize_actors(self):
        config = self.config
        first_vehicle_transform = carla.Transform(
            carla.Location(
                config.other_actors[0].transform.location.x,
                config.other_actors[0].transform.location.y,
                config.other_actors[0].transform.location.z
            ),
            config.other_actors[0].transform.rotation)
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0]

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            raise RuntimeError("No traffic light for the given location found")
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        # all actors just go straight
        for i in range(len(self.other_actors)):
            self.scenario_operation.go_straight(self._target_vel, i)

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionRightTurn(BasicScenario):
    """
        Vehicle turning right at signalized junction scenario an actor has higher priority, ego needs to yield to oncoming actor
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(SignalizedJunctionRightTurn, self).__init__("TurnRightAtSignalizedJunction", config, world)
        self._map = CarlaDataProvider.get_map()
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._target_vel = 12
        self._actor_distance = 100
        self._traffic_light = None
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        # traffic_light_other = CarlaDataProvider.get_next_traffic_light(config.other_actors[0], True)
        if self._traffic_light is None:
            raise RuntimeError("No traffic light for the given location found")
        self._traffic_light.set_state(carla.TrafficLightState.Red)
        self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 35
        self.trigger = False
        self.ego_max_driven_distance = 150

    def initialize_actors(self):
        # create the other vehicle
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
        self.reference_actor = self.other_actors[0]

        # set traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            raise RuntimeError("No traffic light for the given location found")
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        # Actor just move forward with a specific speed
        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self._target_vel, i)

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class NoSignalJunctionCrossingRoute(BasicScenario):
    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(NoSignalJunctionCrossingRoute, self).__init__("NoSignalJunctionCrossingRoute", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self.actor_speed = 10
        self.trigger_distance_threshold = 35
        self.trigger = False
        self.scenario_operation = ScenarioOperation()
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def initialize_actors(self):
        self._other_actor_transform = self.config.other_actors[0].transform
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
        self.reference_actor = self.other_actors[0]

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self.actor_speed, i)

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False
