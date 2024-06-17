''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:49:18
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''


from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_utils import get_valid_spawn_points


class AutopolitBackgroundVehicle(BasicScenario):
    """
        This scenario create background vehicles and control they with autopolit.
        It can be used to train agents under ordinary traffic scenarios.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(AutopolitBackgroundVehicle, self).__init__("AutopolitBackgroundVehicle", config, world)
        self._map = CarlaDataProvider.get_map()
        self.ego_vehicle = ego_vehicle
        self.world = world

        self.timeout = timeout
        self.number_of_vehicles = int(60 / config.num_scenario)
        self.number_of_walkers = 0

    def initialize_actors(self):
        """
        Set other_actors to the superset of all scenario actors
        """
        vehicle_spawn_points = get_valid_spawn_points(self.world)
        count = min(self.number_of_vehicles, len(vehicle_spawn_points))
        for spawn_point in vehicle_spawn_points:
            vehicle = CarlaDataProvider.request_new_actor(
                'vehicle.*', 
                spawn_point=spawn_point, 
                rolename='autopilot',
                autopilot=True, 
                random_location=False
            )
            if vehicle is not None:
                count -= 1
                self.other_actors.append(vehicle)
            if count <= 0:
                break

        # the trigger distance will always be 0, trigger at the beginning
        self.reference_actor = self.ego_vehicle

    def create_behavior(self, scenario_init_action):
        pass

    def update_behavior(self, scenario_action):
        pass

    def check_stop_condition(self):
        return False
