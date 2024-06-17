import json

import numpy as np
import os
import shutil

from utilities import build_route, get_nearist_waypoints, rotate_waypoints, get_map_centers


def create_multi_routes(selected_waypoints, center):
    local_waypoints = selected_waypoints
    # rotate waypoints around the map center
    map_waypoints = []
    for i in range(4):
        theta = i * np.pi / 2
        rotated_waypoints = rotate_waypoints(local_waypoints, center, theta)
        map_waypoints.append(rotated_waypoints)
    map_waypoints = np.vstack(map_waypoints)
    return map_waypoints


def create_route(config, old_center, new_center, selected_waypoints, waypoints_dense):
    # get point shift in x and y locations
    shift = new_center - old_center
    all_routes_waypoints = create_multi_routes(selected_waypoints, old_center)
    all_routes_waypoints[:, :2] += shift.reshape((1, -1))

    # save waypoints
    scenario_id = config.scenario
    save_dir = os.path.join(config.save_dir, f"scenario_{scenario_id:02d}_routes")
    os.makedirs(save_dir, exist_ok=True)

    route_length = len(selected_waypoints)
    route_num = len(all_routes_waypoints) // route_length
    all_routes_real_waypoints = []
    for i in range(route_num):
        route_waypoints = all_routes_waypoints[route_length * i: route_length * (i + 1)]
        real_route_waypoints = []

        # find the closest waypoints from the dense waypoints
        for route_waypoint in route_waypoints:
            idx, dist = get_nearist_waypoints(route_waypoint, waypoints_dense)
            real_route_waypoint = waypoints_dense[idx]
            real_route_waypoints.append(real_route_waypoint)
            yaw_diff = (real_route_waypoint[4] - route_waypoint[4]) % 360
            if dist > 2 or yaw_diff > 10:
                print(f"waypoint {route_waypoint} can not be found on the map, \n"
                      f"assigned to the nearist waypoint {real_route_waypoint}")

        all_routes_real_waypoints.append(np.array(real_route_waypoints))

    return all_routes_real_waypoints


def save_routes(config, save_dir, route):
    route_id = 0
    save_file = os.path.join(save_dir, f"scenario_{config.scenario:02d}_route_{route_id:02d}.xml")
    while os.path.isfile(save_file):
        route_id += 1
        save_file = os.path.join(save_dir, f"scenario_{config.scenario:02d}_route_{route_id:02d}.xml")

    build_route(route, route_id, config.map, save_file)
    return route_id


def main(config):
    np.set_printoptions(suppress=True)
    # all waypoints on the map
    waypoints_dense = np.load(f"map_waypoints/{config.map}/dense.npy")

    # scenario route datas
    scenario_route_datas = []
    data_id = 0

    # get scenario that need to export
    if config.scenario < 0:
        map_dir = os.path.join("scenario_origin", config.map)
        scenarios = list(filter(lambda x: x.endswith("_routes"), os.listdir(map_dir)))
        scenarios = list(map(lambda x: int(x[9: 11]), scenarios))
        scenarios.sort()
    else:
        scenarios = [config.scenario]

    # get centers of the map
    centers = get_map_centers(config.map)

    for scenario in scenarios:
        config.scenario = scenario
        # load scenarios
        scenarios_dir = os.path.join("scenario_origin", config.map, f"scenario_{config.scenario:02d}_routes")
        scenario_file_names = list(filter(lambda x: x.endswith('.npy'), os.listdir(scenarios_dir)))
        scenario_file_names.sort()

        # delete previous exported routes
        save_dir = os.path.join(config.save_dir, f"scenario_{config.scenario:02d}_routes")
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        export_route_num = 0

        for scenario_file_name in scenario_file_names:
            scenario_file = os.path.join(scenarios_dir, scenario_file_name)
            selected_waypoints = np.load(scenario_file)
            # copy waypoints to each center
            for center in centers:
                old_center = centers[0]
                new_center = center
                scenarios_routes = create_route(config, old_center, new_center, selected_waypoints, waypoints_dense)

                # save routes
                for scenarios_route in scenarios_routes:
                    route_id = save_routes(config, save_dir, scenarios_route)

                    # add data to scenario_route_datas
                    data = {
                        "data_id": data_id,
                        "scenario_folder": "standard",
                        "scenario_id": config.scenario,
                        "route_id": route_id,
                        "risk_level": None,
                        "parameters": None
                    }

                    scenario_route_datas.append(data)
                    data_id += 1
                    export_route_num += 1

        print(f"{export_route_num} routes of scenario {scenario} is exported to {scenarios_dir}")

    # save scenario config data
    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, 'standard.json'), 'w') as f:
        json.dump(scenario_route_datas, f, indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Town_Safebench_Light')
    parser.add_argument('--save_dir', type=str, default="scenario_data/route_new_map")
    parser.add_argument('--scenario', type=int, default=-1)

    args = parser.parse_args()

    main(args)