''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-05 15:37:37
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenarioconfigs/scenario_configuration.py>
    Copyright (c) 2019 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla


class ScenarioConfig(object):
    """
        Configuration of parsed scenario
    """

    auto_ego = False
    num_scenario = None
    route_region = ''
    data_id = 0
    scenario_folder = None
    scenario_id = 0
    route_id = 0
    risk_level = 0
    parameters = None

    town = ''
    name = ''
    weather = None
    scenario_file = None
    initial_transform = None
    initial_pose = None
    trajectory = None
    texture_dir = None


class RouteScenarioConfig(object):
    """
        configuration of a RouteScenario
    """
    other_actors = []
    trigger_points = []
    route_var_name = None
    subtype = None
    parameters = None
    weather = carla.WeatherParameters()
    num_scenario = None
    friction = None


class PerceptionScenarioConfig(object):
    """
        configuration of a PerceptionScenario
    """
    other_actors = []
    trigger_points = []
    route_var_name = None
    subtype = None
    parameters = None
    weather = carla.WeatherParameters()
    num_scenario = None
    friction = None
    ego_id = 0
    texture_dir = None


