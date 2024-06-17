''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 19:00:58
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/tools>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import json
import math
import xml.etree.ElementTree as ET

import carla
from agents.navigation.local_planner import RoadOption
from safebench.scenario.scenario_manager.scenario_config import ScenarioConfig


# TODO  check this threshold, it could be a bit larger but not so large that we cluster scenarios.
TRIGGER_THRESHOLD = 2.0  # Threshold to say if a trigger position is new or repeated, works for matching positions
TRIGGER_ANGLE_THRESHOLD = 10  # Threshold to say if two angles can be considering matching when matching transforms.


class RouteParser(object):
    """
        Pure static class used to parse all the route and scenario configuration parameters.
    """

    @staticmethod
    def parse_annotations_file(annotation_filename):
        """
            Return the annotations of which positions where the scenarios are going to happen.
                :param annotation_filename: the filename for the anotations file
                :return:
        """
        with open(annotation_filename, 'r') as f:
            annotation_dict = json.loads(f.read())

        final_dict = {}
        for town_dict in annotation_dict['available_scenarios']:
            final_dict.update(town_dict)
        return final_dict

    @staticmethod
    def parse_routes_file(route_filename, scenario_file, single_route=None):
        """
            Returns a list of route elements.
                :param route_filename: the path to a set of routes.
                :param single_route: If set, only this route shall be returned
                :return: List of dicts containing the waypoints, id and town of the routes
        """

        list_route_descriptions = []
        tree = ET.parse(route_filename)
        for route in tree.iter("route"):
            route_id = route.attrib['id']
            if single_route and route_id != single_route:
                continue

            new_config = ScenarioConfig()
            new_config.town = route.attrib['town']
            new_config.route_region = route.attrib['region'] if 'region' in route.attrib else None
            new_config.name = "RouteScenario_{}".format(route_id)
            new_config.weather = RouteParser.parse_weather(route)
            new_config.scenario_file = scenario_file

            waypoint_list = []  # the list of waypoints that can be found on this route
            for waypoint in route.iter('waypoint'):
                if len(waypoint_list) == 0:
                    pitch = float(waypoint.attrib['pitch'])
                    roll = float(waypoint.attrib['roll'])
                    yaw = float(waypoint.attrib['yaw'])
                    x = float(waypoint.attrib['x'])
                    y = float(waypoint.attrib['y'])
                    z = float(waypoint.attrib['z']) + 2.0  # avoid collision to the ground
                    initial_pose = carla.Transform(carla.Location(x, y, z), carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))
                    new_config.initial_transform = initial_pose
                    new_config.initial_pose = initial_pose
                waypoint_list.append(carla.Location(x=float(waypoint.attrib['x']), y=float(waypoint.attrib['y']), z=float(waypoint.attrib['z'])))

            new_config.trajectory = waypoint_list
            list_route_descriptions.append(new_config)

        return list_route_descriptions

    @staticmethod
    def parse_weather(route):
        """
            Returns a carla.WeatherParameters with the corresponding weather for that route. 
            If the route has no weather attribute, the default one is triggered.
        """

        route_weather = route.find("weather")
        if route_weather is None:
            weather = carla.WeatherParameters(sun_altitude_angle=70)
        else:
            weather = carla.WeatherParameters()
            for weather_attrib in route.iter("weather"):
                if 'cloudiness' in weather_attrib.attrib:
                    weather.cloudiness = float(weather_attrib.attrib['cloudiness'])
                if 'precipitation' in weather_attrib.attrib:
                    weather.precipitation = float(weather_attrib.attrib['precipitation'])
                if 'precipitation_deposits' in weather_attrib.attrib:
                    weather.precipitation_deposits = float(weather_attrib.attrib['precipitation_deposits'])
                if 'wind_intensity' in weather_attrib.attrib:
                    weather.wind_intensity = float(weather_attrib.attrib['wind_intensity'])
                if 'sun_azimuth_angle' in weather_attrib.attrib:
                    weather.sun_azimuth_angle = float(weather_attrib.attrib['sun_azimuth_angle'])
                if 'sun_altitude_angle' in weather_attrib.attrib:
                    weather.sun_altitude_angle = float(weather_attrib.attrib['sun_altitude_angle'])
                if 'wetness' in weather_attrib.attrib:
                    weather.wetness = float(weather_attrib.attrib['wetness'])
                if 'fog_distance' in weather_attrib.attrib:
                    weather.fog_distance = float(weather_attrib.attrib['fog_distance'])
                if 'fog_density' in weather_attrib.attrib:
                    weather.fog_density = float(weather_attrib.attrib['fog_density'])
        return weather

    @staticmethod
    def check_trigger_position(new_trigger, existing_triggers):
        """
            Check if this trigger position already exists or if it is a new one.
                :param new_trigger:
                :param existing_triggers:
                :return:
        """
        for trigger_id in existing_triggers.keys():
            trigger = existing_triggers[trigger_id]
            dx = trigger['x'] - new_trigger['x']
            dy = trigger['y'] - new_trigger['y']
            distance = math.sqrt(dx * dx + dy * dy)

            dyaw = (trigger['yaw'] - new_trigger['yaw']) % 360
            if distance < TRIGGER_THRESHOLD and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD)):
                return trigger_id
        return None

    @staticmethod
    def convert_waypoint_float(waypoint):
        waypoint['x'] = float(waypoint['x'])
        waypoint['y'] = float(waypoint['y'])
        waypoint['z'] = float(waypoint['z'])
        waypoint['yaw'] = float(waypoint['yaw'])

    @staticmethod
    def match_world_location_to_route(world_location, route_description):
        """
            We match this location to a given route.
                world_location:
                route_description:
        """
        def match_waypoints(waypoint1, wtransform):
            """
            Check if waypoint1 and wtransform are similar
            """
            dx = float(waypoint1['x']) - wtransform.location.x
            dy = float(waypoint1['y']) - wtransform.location.y
            dz = float(waypoint1['z']) - wtransform.location.z
            dpos = math.sqrt(dx * dx + dy * dy + dz * dz)

            dyaw = (float(waypoint1['yaw']) - wtransform.rotation.yaw) % 360
            return dpos < TRIGGER_THRESHOLD and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD))

        match_position = 0
        # TODO this function can be optimized to run on Log(N) time
        for route_waypoint in route_description:
            if match_waypoints(world_location, route_waypoint[0]):
                return match_position
            match_position += 1

        return None

    @staticmethod
    def scan_route_for_scenarios(route_name, trajectory, world_annotations, scenario_id=None):
        """
            Returns a plain list of possible scenarios that can happen in this route 
            by matching the locations from the scenario into the route description
        """

        # the triggers dictionaries
        existent_triggers = {}

        # We have a table of IDs and trigger positions associated
        possible_scenarios = {}

        # Keep track of the trigger ids being added
        latest_trigger_id = 0

        for town_name in world_annotations.keys():
            if town_name != route_name:
                continue

            triggers = []
            matched_triggers = []

            scenarios = world_annotations[town_name]
            for scenario in scenarios:  # For each existent scenario
                if "scenario_name" not in scenario:
                    raise ValueError('Scenario type not found in scenario description')

                scenario_name = scenario["scenario_name"]
                for event in scenario["available_event_configurations"]:
                    waypoint = event['transform']  # trigger point of this scenario
                    if scenario_id == 0:
                        waypoint = {
                            "pitch": trajectory[0][0].rotation.pitch,
                            "x": trajectory[0][0].location.x,
                            "y": trajectory[0][0].location.y,
                            "yaw": trajectory[0][0].rotation.yaw,
                            "z": trajectory[0][0].location.z
                        }
                    RouteParser.convert_waypoint_float(waypoint)
                    
                    # we match trigger point to the route, now we need to check if the route affects
                    triggers.append([waypoint['x'], waypoint['y'], waypoint['z']])

                    match_position = RouteParser.match_world_location_to_route(waypoint, trajectory)
                    if scenario_id == 0:
                        assert match_position == 0
                    if match_position is not None:
                        matched_triggers.append([waypoint['x'], waypoint['y'], waypoint['z']])

                        # We match a location for this scenario, create a scenario object so this scenario can be instantiated later
                        other_vehicles = None
                        if 'other_actors' in event:
                            other_vehicles = event['other_actors']

                        scenario_description = {
                            'name': scenario_name,
                            'other_actors': other_vehicles,
                            'trigger_position': waypoint,
                            'match_position': match_position,
                        }

                        trigger_id = RouteParser.check_trigger_position(waypoint, existent_triggers)
                        if trigger_id is None:
                            # This trigger does not exist create a new reference on existent triggers
                            existent_triggers.update({latest_trigger_id: waypoint})
                            # Update a reference for this trigger on the possible scenarios
                            possible_scenarios.update({latest_trigger_id: []})
                            trigger_id = latest_trigger_id
                            # Increment the latest trigger
                            latest_trigger_id += 1
                        possible_scenarios[trigger_id].append(scenario_description)
        
        return possible_scenarios, existent_triggers

    @staticmethod
    def match_route_and_scenarios(town_name, trajectory, possible_scenarios, scenario_id=None):
        select_scenarios = []
        for town_name in possible_scenarios.keys():
            if town_name != town_name:
                continue

            scenarios = possible_scenarios[town_name]
            for scenario in scenarios: 
                # scenario must have scenario type
                if "scenario_name" not in scenario:
                    raise ValueError('Scenario type not found in scenario description')

                scenario_name = scenario["scenario_name"]
                for event in scenario["available_event_configurations"]:
                    waypoint = event['transform']  # trigger point of this scenario
                    if scenario_id == 0:
                        waypoint = {
                            "pitch": trajectory[0][0].rotation.pitch,
                            "x": trajectory[0][0].location.x,
                            "y": trajectory[0][0].location.y,
                            "yaw": trajectory[0][0].rotation.yaw,
                            "z": trajectory[0][0].location.z
                        }
                    RouteParser.convert_waypoint_float(waypoint)

                    # We match a location for this scenario, create a scenario object so this scenario can be instantiated later
                    other_vehicles = None
                    if 'other_actors' in event:
                        other_vehicles = event['other_actors']
                    scenario_description = {
                        'name': scenario_name,
                        'other_actors': other_vehicles,
                        'trigger_position': waypoint,
                    }
                    select_scenarios.append(scenario_description)
        return select_scenarios