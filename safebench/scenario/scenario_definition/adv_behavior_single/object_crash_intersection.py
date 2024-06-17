''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 17:32:41
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import math

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_helper import get_crossing_point, get_junction_topology


def get_opponent_transform(added_dist, waypoint, trigger_location):
    """
        Calculate the transform of the adversary
    """
    lane_width = waypoint.lane_width

    offset = {"orientation": 270, "position": 90, "k": 1.0}
    _wp = waypoint.next(added_dist)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    location = _wp.transform.location
    orientation_yaw = _wp.transform.rotation.yaw + offset["orientation"]
    position_yaw = _wp.transform.rotation.yaw + offset["position"]

    offset_location = carla.Location(
        offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
        offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
    location += offset_location
    location.x = trigger_location.x + 20
    location.z = trigger_location.z
    transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))

    return transform


def get_right_driving_lane(waypoint):
    """
        Gets the driving / parking lane that is most to the right of the waypoint as well as the number of lane changes done
    """
    lane_changes = 0
    while True:
        wp_next = waypoint.get_right_lane()
        lane_changes += 1

        if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
            break
        elif wp_next.lane_type == carla.LaneType.Shoulder:
            # Filter Parkings considered as Shoulders
            if is_lane_a_parking(wp_next):
                lane_changes += 1
                waypoint = wp_next
            break
        else:
            waypoint = wp_next
    return waypoint, lane_changes


def is_lane_a_parking(waypoint):
    """
        This function filters false negative Shoulder which are in reality Parking lanes.
        These are differentiated from the others because, similar to the driving lanes,
        they have, on the right, a small Shoulder followed by a Sidewalk.
    """

    # Parking are wide lanes
    if waypoint.lane_width > 2:
        wp_next = waypoint.get_right_lane()

        # That are next to a mini-Shoulder
        if wp_next is not None and wp_next.lane_type == carla.LaneType.Shoulder:
            wp_next_next = wp_next.get_right_lane()

            # Followed by a Sidewalk
            if wp_next_next is not None and wp_next_next.lane_type == carla.LaneType.Sidewalk:
                return True
    return False


class VehicleTurningRoute(BasicScenario):
    """
        The ego vehicle is passing through a road and encounters a cyclist after taking a turn. 
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(VehicleTurningRoute, self).__init__("VehicleTurningRoute-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self.running_distance = 10
        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 20
        self.ego_max_driven_distance = 180

    def convert_actions(self, actions, x_scale, y_scale, x_mean, y_mean):

        x = x_mean
        y = y_mean
        yaw = yaw_mean
        dist = dist_mean
        return [x, y, yaw, dist]

    def convert_actions(self, actions):
        base_speed = 5.0
        speed_scale = 5.0
        speed = actions[0] * speed_scale + base_speed
        return speed

    def initialize_actors(self):
        cross_location = get_crossing_point(self.ego_vehicle)
        cross_waypoint = CarlaDataProvider.get_map().get_waypoint(cross_location)
        entry_wps, exit_wps = get_junction_topology(cross_waypoint.get_junction())
        assert len(entry_wps) == len(exit_wps)
        x_mean = y_mean = 0
        max_x_scale = max_y_scale = 0
        for i in range(len(entry_wps)):
            x_mean += entry_wps[i].transform.location.x + exit_wps[i].transform.location.x
            y_mean += entry_wps[i].transform.location.y + exit_wps[i].transform.location.y
        x_mean /= len(entry_wps) * 2
        y_mean /= len(entry_wps) * 2
        for i in range(len(entry_wps)):
            max_x_scale = max(max_x_scale, abs(entry_wps[i].transform.location.x - x_mean), abs(exit_wps[i].transform.location.x - x_mean))
            max_y_scale = max(max_y_scale, abs(entry_wps[i].transform.location.y - y_mean), abs(exit_wps[i].transform.location.y - y_mean))
        max_x_scale *= 0.8
        max_y_scale *= 0.8

        x = x_mean
        y = y_mean
        yaw = 180
        other_actor_transform = carla.Transform(carla.Location(x, y, 0), carla.Rotation(yaw=yaw))
        
        self.actor_transform_list = [other_actor_transform]
        self.actor_type_list = ['vehicle.diamondback.century']
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario
        
    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] initial action.'

    def update_behavior(self, scenario_action):
        cur_actor_target_speed = self.convert_actions(scenario_action)
        self.scenario_operation.go_straight(cur_actor_target_speed, 0)

    def check_stop_condition(self):
        pass
