import json

import numpy as np
import os
import matplotlib.pyplot as pl

from utilities import build_scenarios, get_nearist_waypoints, rotate_waypoints, get_map_centers


def create_multi_scenarios(selected_waypoints, center):
    local_waypoints = selected_waypoints
    # rotate waypoints around the map center
    map_waypoints = []
    for i in range(4):
        theta = i * np.pi / 2
        rotated_waypoints = rotate_waypoints(local_waypoints, center, theta)
        map_waypoints.append(rotated_waypoints)
    map_waypoints = np.vstack(map_waypoints)
    return map_waypoints


def create_scenario(config, center, center_new, selected_waypoints, waypoints_dense):
    # get point shift in x and y locations
    shift = center_new - center
    all_scenarios_waypoints = create_multi_scenarios(selected_waypoints, center)
    all_scenarios_waypoints[:, :2] += shift.reshape((1, -1))

    # save waypoints
    scenario_length = len(selected_waypoints)
    scenario_num = len(all_scenarios_waypoints) // scenario_length

    all_scenarios_configs = []
    for i in range(scenario_num):
        scenario_waypoints = all_scenarios_waypoints[scenario_length * i: scenario_length * (i + 1)]
        real_scenario_waypoints = []

        # find the closest waypoints from the dense waypoints
        for scenario_waypoint in scenario_waypoints:
            idx, dist = get_nearist_waypoints(scenario_waypoint, waypoints_dense)
            real_scenario_waypoint = waypoints_dense[idx]
            real_scenario_waypoints.append(real_scenario_waypoint)
            yaw_diff = (real_scenario_waypoint[4] - scenario_waypoint[4]) % 360
            if dist > 2 or yaw_diff > 10:
                print(f"waypoint {scenario_waypoint} can not be found on the map, assigned to the nearist waypoint {real_scenario_waypoint}")

        scenario_config = build_scenarios(real_scenario_waypoints)
        all_scenarios_configs.append(scenario_config)

    return all_scenarios_configs


def save_scenarios(config, scenarios_configs):
    scenario_id = config.scenario
    save_dir = os.path.join(config.save_dir, f"scenarios")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"scenario_{scenario_id:02d}.json")
    scenario_json = {
        "available_scenarios": [
            {
                config.map: [
                    {
                        "available_event_configurations": scenarios_configs,
                        "scenario_name": f"Scenario{scenario_id + 2}"
                    }
                ]
            }
        ]
    }
    with open(save_file, 'w') as f:
        json.dump(scenario_json, f, indent=2)


def main(config):
    np.set_printoptions(suppress=True)
    # all waypoints on the map
    waypoints_dense = np.load(f"map_waypoints/{config.map}/dense.npy")

    # get map centers
    centers = get_map_centers(config.map)

    # get scenario that need to export
    if config.scenario < 0:
        map_dir = os.path.join("scenario_origin", config.map)
        scenarios = list(filter(lambda x: x.endswith("_scenarios"), os.listdir(map_dir)))
        scenarios = list(map(lambda x: int(x[9: 11]), scenarios))
        scenarios.sort()
    else:
        scenarios = [config.scenario]

    for scenario in scenarios:
        config.scenario = scenario
        # load scenarios
        all_scenario_configs = []
        save_dir = os.path.join("scenario_origin", config.map, f"scenario_{config.scenario:02d}_scenarios")
        scenario_file_names = os.listdir(save_dir)
        scenario_file_names.sort()
        for scenario_file_name in scenario_file_names:
            scenario_file = os.path.join(save_dir, scenario_file_name)
            selected_waypoints = np.load(scenario_file)
            for center in centers:
                scenarios_configs = create_scenario(config, centers[0], center, selected_waypoints, waypoints_dense)
                all_scenario_configs += scenarios_configs

        # save scenarios
        save_scenarios(config, all_scenario_configs)
        print(f"{len(all_scenario_configs)} scenarios of scenario {scenario} is exported to {save_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Town_Safebench_Light')
    parser.add_argument('--save_dir', type=str, default="scenario_data/route_new_map")
    parser.add_argument('--scenario', type=int, default=-1)

    args = parser.parse_args()

    main(args)

