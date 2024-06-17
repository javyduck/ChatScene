''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-07 01:26:56
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenariomanager/scenario_manager.py>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.tools.scenario_utils import calculate_distance_locations


class ScenarioManager(object):
    """
        Dynamic version scenario manager class. This class holds all functionality
        required to initialize, trigger, update and stop a scenario.
    """

    def __init__(self, logger, use_scenic=False):
        self.logger = logger
        self.scenic = use_scenic
        self._reset()

    def _reset(self):
        #self.scenario = None
        self.background_scenario = None
        self.ego_vehicle = None
        self.scenario_list = None
        self.triggered_scenario = set()
        self._running = False
        self._timestamp_last_run = 0.0
        self.running_record = []
        GameTime.restart()

    def clean_up(self):
        if self.background_scenario is not None:
            self.background_scenario.clean_up()

    def load_scenario(self, scenario):
        self._reset()
        self.background_scenario = scenario
        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_list = scenario.list_scenarios

    def run_scenario(self, scenario_init_action):
        self._running = True
        self._init_scenarios(scenario_init_action)

    def _init_scenarios(self, scenario_init_action):
        # spawn background actors
        self.background_scenario.initialize_actors()
        
        # spawn actors for each scenario along this route
        for running_scenario in self.scenario_list:
            # some scenario passes actions when creating behavior
            running_scenario.create_behavior(scenario_init_action)
            # init actors after passing in init actions
            running_scenario.initialize_actors()
    
    def stop_scenario(self):
        self._running = False

    def update_running_status(self):
        record, stop = self.background_scenario.get_running_status(self.running_record)
        self.running_record.append(record)
        if stop:
            self._running = False

    def get_update(self, timestamp, scenario_action):
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            
            if self.scenic:
                # update behavior of triggered scenarios
                for running_scenario in self.scenario_list: 
                    # update behavior of agents in scenario
                    running_scenario.update_behavior(scenario_action)
            else:
                # check whether the scenario should be triggered
                for spawned_scenario in self.scenario_list:
                    ego_location = CarlaDataProvider.get_location(self.ego_vehicle)
                    cur_distance = None
                    reference_location = None
                    if spawned_scenario.reference_actor:
                        reference_location = CarlaDataProvider.get_location(spawned_scenario.reference_actor)
                    if reference_location:
                        cur_distance = calculate_distance_locations(ego_location, reference_location)

                    if cur_distance and cur_distance < spawned_scenario.trigger_distance_threshold:
                        if spawned_scenario not in self.triggered_scenario:
                            self.logger.log(">> Trigger scenario: " + spawned_scenario.name)
                            self.triggered_scenario.add(spawned_scenario)

                # update behavior of triggered scenarios
                for running_scenario in self.triggered_scenario: 
                    # update behavior of agents in scenario
                    running_scenario.update_behavior(scenario_action)

            self.update_running_status()