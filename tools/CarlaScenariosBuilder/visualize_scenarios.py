import numpy as np
import os
import json
import matplotlib.pyplot as plt
from utilities import parse_scenarios, select_waypoints, get_view_centers, get_map_centers


def draw(ax, map_name, scenario_config, centers, waypoints_sparse, zoom):
    if zoom:
        draw_zoom(ax, scenario_config, centers, waypoints_sparse)
    else:
        draw_global(ax, map_name, scenario_config, waypoints_sparse)


def draw_global(ax, map_name, scenario_config, waypoints_sparse):
    ax.cla()
    ax.plot(waypoints_sparse[:, 0], -waypoints_sparse[:, 1], 'o', color='y')

    waypoints = parse_scenarios(scenario_config)

    start_waypoint = waypoints[0]
    for end_waypoint in waypoints[1:]:
        ax.plot([start_waypoint[0], end_waypoint[0]], [-start_waypoint[1], -end_waypoint[1]], '--', color='b')
        ax.plot(end_waypoint[0], -end_waypoint[1], 'o', color='g')
        ax.text(end_waypoint[0] + 8, -end_waypoint[1] + 8, "Actor", bbox=dict(facecolor='green', alpha=0.7))
    ax.plot(start_waypoint[0], -start_waypoint[1], 'o', color='r')
    ax.text(start_waypoint[0] + 12, -start_waypoint[1] + 8, "Trigger", bbox=dict(facecolor='red', alpha=0.7))

    centers = get_map_centers(map_name)
    if centers is None:
        center = waypoints_sparse[:, :2].mean(0)
    else:
        center = centers.mean(0)

    dist = np.linalg.norm(waypoints_sparse[:, :2] - center, np.inf, axis=1).max() + 20
    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def draw_zoom(ax, scenario_config, centers, waypoints_sparse):
    waypoints = parse_scenarios(scenario_config)

    geo_center = waypoints[:, :2].mean(0)
    if centers is None:
        center = geo_center
    else:
        center_dists = np.linalg.norm(np.array(centers) - geo_center, axis=1)
        center = centers[center_dists.argmin()]
    dist = 120

    ax.cla()
    road_waypoints = select_waypoints(waypoints_sparse, center, dist)
    ax.plot(road_waypoints[:, 0], -road_waypoints[:, 1], 'o', color='y')
    for road_waypoint in road_waypoints:
        length = 4
        x, y, z, pitch, yaw, roll = road_waypoint
        yaw = yaw / 180 * np.pi
        dx, dy = length * np.cos(yaw), length * np.sin(yaw)
        plt.arrow(x, -y, dx, -dy, color='y')

    start_waypoint = waypoints[0]
    length = 10
    for end_waypoint in waypoints[1:]:
        ax.plot([start_waypoint[0], end_waypoint[0]], [-start_waypoint[1], -end_waypoint[1]], '--', color='b')
        ax.plot(end_waypoint[0], -end_waypoint[1], 'o', color='g')
        ax.text(end_waypoint[0] + 8, -end_waypoint[1] + 8, "Actor", bbox=dict(facecolor='green', alpha=0.7))

        # print("end_waypoint", end_waypoint)
        x, y, z, pitch, yaw, roll = end_waypoint
        yaw = yaw / 180 * np.pi
        dx, dy = length * np.cos(yaw), length * np.sin(yaw)
        plt.arrow(x, -y, dx, -dy, color='g')


    ax.plot(start_waypoint[0], -start_waypoint[1], 'o', color='r')
    ax.text(start_waypoint[0] + 12, -start_waypoint[1] + 8, "Trigger", bbox=dict(facecolor='red', alpha=0.7))

    dist = 120
    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def set_title(ax, map_name, scenario_idx, title=None):
    if title is None:
        title = "Left and right click to change scenario.\nMiddle click to zoom."
    title = f"Visualizing scenario: {scenario_idx} of map {map_name}.\n" + title
    ax.set_title(title, fontsize=12, loc='left')


def main(config):
    # get scenarios
    scenario_id = config.scenario
    scenario_dir = os.path.join(config.save_dir, f"scenarios/scenario_{scenario_id:02d}.json")
    with open(scenario_dir, 'r') as f:
        scenario_json = json.load(f)

    scenario_configs = []
    for available_scenarios in scenario_json["available_scenarios"]:
        for map_name in available_scenarios:
            for available_event_configurations in available_scenarios[map_name]:
                for event_configuration in available_event_configurations["available_event_configurations"]:
                    if config.map == "None" or config.map == map_name:
                        event_configuration["map"] = map_name
                        scenario_configs.append(event_configuration)
    # map_scenario_configs = scenario_json["available_scenarios"][0][config.map][0]["available_event_configurations"]
    # print(scenario_configs[0])
    # quit()
    scenario_num = len(scenario_configs)
    scenario_idx = 0
    zoom = False
    map_name = scenario_configs[scenario_idx]["map"]

    # sparse waypoints used for visualization
    waypoints_sparse = np.load(f"map_waypoints/{map_name}/sparse.npy")

    # initialize potential center position
    centers = get_view_centers(map_name)

    assert scenario_num > 0, "No scenario to visualize."

    def onclick(event):
        nonlocal scenario_idx, zoom, map_name, waypoints_sparse, centers

        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))

        new_scenario_id = scenario_idx
        if int(event.button) == 1:
            # left click to visualize next scenario
            new_scenario_id = (scenario_idx + 1) % scenario_num
        elif int(event.button) == 3:
            # right click to visualize previous scenario
            new_scenario_id = (scenario_idx - 1) % scenario_num
        elif int(event.button) == 2:
            # middle click to zoom
            zoom = not zoom

        if new_scenario_id != scenario_idx:
            scenario_idx = new_scenario_id

        if 1 <= int(event.button) <= 3:
            if map_name != scenario_configs[scenario_idx]['map']:
                map_name = scenario_configs[scenario_idx]['map']

                # sparse waypoints used for visualization
                waypoints_sparse = np.load(f"map_waypoints/{map_name}/sparse.npy")
                # initialize potential center position
                centers = get_view_centers(map_name)

            draw(ax, map_name, scenario_configs[scenario_idx], centers, waypoints_sparse, zoom)
            set_title(ax, map_name, scenario_idx)
            plt.draw()

    # visualize waypoints
    fig, ax = plt.subplots(figsize=(12, 12))
    draw(ax, map_name, scenario_configs[scenario_idx], centers, waypoints_sparse, zoom)
    set_title(ax, map_name, scenario_idx)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default="None")
    parser.add_argument('--save_dir', type=str, default="scenario_data/route_new_map")
    parser.add_argument('--scenario', type=int, required=True)


    args = parser.parse_args()

    main(args)
