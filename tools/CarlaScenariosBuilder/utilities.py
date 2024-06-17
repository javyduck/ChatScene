import numpy as np
import os
import xml.etree.cElementTree as ET
import shutil


def build_route(waypoints, route_id, town, save_file):
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", id=f'{route_id}', town=town)

    ET.SubElement(
        route,
        "weather",
        cloudiness="0",
        precipitation="0",
        precipitation_deposits="0",
        wind_intensity="0",
        sun_azimuth_angle="0",
        sun_altitude_angle="70",
        fog_density="0",
        fog_distance="0",
        wetness="0", )

    for waypoint in waypoints:
        x, y, z, pitch, yaw, roll = waypoint
        ET.SubElement(
            route,
            "waypoint",
            pitch=f"{pitch:.2f}",
            roll=f"{roll:.2f}",
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            yaw=f"{yaw:.2f}",
            z=f"{z:.2f}"
        )

    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(save_file, encoding='utf-8', xml_declaration=True)


def build_scenarios(waypoints):
    scenario_waypoints = []
    for waypoint in waypoints:
        x, y, z, pitch, yaw, roll = waypoint
        point = {
            "pitch": f'{pitch:.2f}',
            "x": f'{x:.2f}',
            "y": f'{y:.2f}',
            "yaw": f'{yaw:.2f}',
            "z": f'{z:.2f}'
        }
        scenario_waypoints.append(point)

    # The first waypoint is used as the trigger point, rest are used as actor spawn points
    config = {
        "other_actors": {
            "left": scenario_waypoints[1:]
        },
        "transform": scenario_waypoints[0]
    }
    return config


def parse_route(route_file):
    tree = ET.parse(route_file)
    waypoints = []
    map_names = []
    for route in tree.iter("route"):
        map_name = route.attrib['town']
        map_names.append(map_name)
        waypoint_list = []  # the list of waypoints that can be found on this route
        for waypoint in route.iter('waypoint'):
            pitch = float(waypoint.attrib['pitch'])
            roll = float(waypoint.attrib['roll'])
            yaw = float(waypoint.attrib['yaw'])
            x = float(waypoint.attrib['x'])
            y = float(waypoint.attrib['y'])
            z = float(waypoint.attrib['z'])
            waypoint_list.append([x, y, z, pitch, yaw, roll])
        waypoints.append(waypoint_list)
    return np.asarray(waypoints), map_names


def parse_scenarios(scenario_config):
    waypoints = []
    trigger_point = scenario_config["transform"]
    x, y, z = float(trigger_point["x"]), float(trigger_point["y"]), float(trigger_point["z"])
    pitch, yaw = float(trigger_point["pitch"]), float(trigger_point["yaw"])
    roll = 0.0
    waypoints.append([x, y, z, pitch, yaw, roll])

    if "other_actors" in scenario_config:
        for actor_type in ["left", "right", "front"]:
            if actor_type in scenario_config["other_actors"]:
                for actor_config in scenario_config["other_actors"][actor_type]:
                    x, y, z = float(actor_config["x"]), float(actor_config["y"]), float(actor_config["z"])
                    pitch, yaw = float(actor_config["pitch"]), float(actor_config["yaw"])
                    roll = 0.0
                    waypoints.append([x, y, z, pitch, yaw, roll])
    return np.asarray(waypoints)


def select_waypoints(waypoints, center, distance):
    # select waypoints around the center within a distance
    mask = np.linalg.norm(waypoints[:, :2] - center, np.inf, axis=1) < distance
    waypoints = waypoints[mask]
    return waypoints


def rotate_waypoints(origin_waypoints, center, theta):
    # waypoint format: x, y, z, pitch, yaw, roll
    m = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    center = np.array(center).reshape((1, 2))
    rotated_waypoints = origin_waypoints.copy()
    rotated_waypoints[:, :2] = (origin_waypoints[:, :2] - center) @ m.T + center
    # change the yaw angle
    rotated_waypoints[:, 4] = (rotated_waypoints[:, 4] + theta / np.pi * 180) % 360
    return rotated_waypoints


def get_nearist_waypoints(waypoint, waypoints):
    waypoint = waypoint.reshape((1, -1))
    waypoints_dist = np.linalg.norm(waypoints[:, :2] - waypoint[:, :2], axis=1)
    return waypoints_dist.argmin(), waypoints_dist.min()


def _get_batch_centers(x_num, y_num):
    map_size = 440
    centers = []
    for j in range(x_num):
        for i in range(y_num):
            centers.append([i, j])
    centers = np.asarray(centers) * map_size
    centers = centers - centers.mean(0)
    return centers


def get_map_centers(map_name):
    if map_name == "Town_Safebench_Light" or map_name == 'town_4intersection_2lane':
        centers = np.asarray([[100, 100]])
    elif map_name == "town_4intersection_2lane_4x4":
        centers = _get_batch_centers(4, 4)
    elif map_name == "town_4intersection_2lane_3x3":
        centers = _get_batch_centers(3, 3)
    elif map_name == "town_4intersection_2lane_2x4":
        centers = _get_batch_centers(2, 4)
    elif map_name == "town_4intersection_2lane_2x2":
        centers = _get_batch_centers(2, 2)
    elif map_name == "town_4intersection_2lane_1x1":
        centers = _get_batch_centers(1, 1)
    else:
        return None
        # raise ValueError(f"Map {map_name} is not supported.")
    return centers


def get_view_centers(map_name):
    map_centers = get_map_centers(map_name)
    if map_centers is None:
        return None

    view_centers = []
    for map_center in map_centers:
        local_centers = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    # ignore the map center
                    continue
                local_centers.append([i * 100, j * 100])
        centers = np.asarray(local_centers) + map_center
        view_centers.append(centers)
    return np.vstack(view_centers)


def copy_routes_and_scenarios(old_map_name, new_map_name):
    old_map_dir = f"scenario_origin/{old_map_name}"
    new_map_dir = f"scenario_origin/{new_map_name}"
    action = input(f"Careful! This action will removed all files in {new_map_dir}. You you want to pressed? [yes/no]\n")
    if os.path.isdir(new_map_dir) and action == 'yes':
        shutil.rmtree(new_map_dir)

    coord_shift = get_map_centers(new_map_name)[0: 1] - get_map_centers(old_map_name)[0: 1]
    for dir_path, dir_names, file_names in os.walk(old_map_dir):
        for file_name in file_names:
            if file_name.endswith('.npy'):
                old_file_path = os.path.join(dir_path, file_name)
                new_dir = dir_path.replace(old_map_dir, new_map_dir)
                new_file_path = os.path.join(new_dir, file_name)
                os.makedirs(new_dir, exist_ok=True)
                waypoints = np.load(old_file_path)
                waypoints[:, :2] += coord_shift
                np.save(new_file_path, waypoints)
    print(f"copy scenarios from {old_map_name} to {new_map_name}")


if __name__ == '__main__':
    copy_routes_and_scenarios("town_4intersection_2lane", "Town_Safebench_Light")
    # import matplotlib.pyplot as plt
    #
    # a = np.load('map_waypoints/town_4intersection_2lane/sparse.npy')
    # plt.scatter(a[:, 0], a[:, 1], color='b')
    #
    # a = get_view_centers("town_4intersection_2lane")
    # plt.scatter(a[:, 0], a[:, 1], color='r')
    # plt.show()





