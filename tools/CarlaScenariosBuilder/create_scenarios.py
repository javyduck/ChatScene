import numpy as np
import os
import matplotlib.pyplot as plt

from utilities import select_waypoints, get_nearist_waypoints, get_map_centers


def draw(ax, center, dist, road_waypoints, selected_waypoints_idx):
    ax.cla()
    ax.plot(road_waypoints[:, 0], -road_waypoints[:, 1], 'o', color='y')

    for road_waypoint in road_waypoints:
        length = 4
        x, y, z, pitch, yaw, roll = road_waypoint
        yaw = yaw / 180 * np.pi
        dx, dy = length * np.cos(yaw), length * np.sin(yaw)
        plt.arrow(x, -y, dx, -dy, color='y')

    if len(selected_waypoints_idx) > 0:
        waypoints_idx = np.array(selected_waypoints_idx)
        waypoints = np.take(road_waypoints, waypoints_idx, axis=0)

        start_waypoint = waypoints[0]
        for end_waypoint in waypoints[1:]:
            ax.plot([start_waypoint[0], end_waypoint[0]], [-start_waypoint[1], -end_waypoint[1]], '--', color='b')
            ax.plot(end_waypoint[0], -end_waypoint[1], 'o', color='g')
            ax.text(end_waypoint[0] + 8, -end_waypoint[1] + 8, "Actor", bbox=dict(facecolor='green', alpha=0.7))
        ax.plot(start_waypoint[0], -start_waypoint[1], 'o', color='r')
        ax.text(start_waypoint[0] + 12, -start_waypoint[1] + 8, "Trigger", bbox=dict(facecolor='red', alpha=0.7))

    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def set_title(ax, title=None):
    if title is None:
        title = "Left click:  select or remove waypoints.\nRight click:  save scenarios to file."
    ax.set_title(title, fontsize=25, loc='left')


def load_scenario(config, dist, waypoints_sparse, save_dir):
    # load route if route is set
    center = None
    road_waypoints = None
    selected_waypoints_idx = []
    if config.scenario_idx >= 0:
        # check if route is created
        scenario_file = os.path.join(save_dir, f'scenario_{config.scenario_idx:02d}.npy')
        if os.path.isfile(scenario_file):
            # load existing routes
            selected_waypoints = np.load(scenario_file)

            # change world center based on waypoints
            geo_center = selected_waypoints[:, :2].mean(0)

            centers = get_map_centers(config.map)[0]
            centers = np.asarray([centers + [-100, -100], centers + [0, -100]])

            center_dists = np.linalg.norm(np.array(centers) - geo_center, axis=1)
            center = centers[center_dists.argmin()]

            # select waypoints
            road_waypoints = select_waypoints(waypoints_sparse, center, dist)
            for waypoint in selected_waypoints:
                idx, dist = get_nearist_waypoints(waypoint, road_waypoints)
                if dist > 5:
                    print(f"waypoint {waypoint} can not be found on the map, "
                          f"assigned to the nearist waypoint {road_waypoints[idx]}")
                selected_waypoints_idx.append(idx)

    return center, road_waypoints, selected_waypoints_idx


def save_scenario(config, save_dir, selected_waypoints):
    scenario_idx = config.scenario_idx
    if scenario_idx < 0:
        # create new route
        scenario_idx = 0
        while os.path.isfile(os.path.join(save_dir, f'scenario_{scenario_idx:02d}.npy')):
            scenario_idx += 1

    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'scenario_{scenario_idx:02d}.npy')
    np.save(save_file, selected_waypoints)


def main(config):
    # sparse waypoints used for user to select
    waypoints_sparse = np.load(f"map_waypoints/{config.map}/sparse.npy")
    # waypoints selected by user
    selected_waypoints_idx = []

    # set road type if is not set
    if config.road == 'auto':
        if config.scenario + 2 in [1, 2, 3, 5, 6]:
            config.road = 'straight'
        elif config.scenario + 2 in [4, 7, 8, 9, 10]:
            config.road = 'intersection'
        else:
            raise ValueError("scenario can not be found.")

    # set waypoints that user can work on
    center = get_map_centers(config.map)[0]
    dist = 120
    if config.road == 'intersection':
        center = center + [-100, -100]
    else:
        center = center + [0, -100]
    road_waypoints = select_waypoints(waypoints_sparse, center, dist)

    # load routes
    scenario_id = config.scenario
    save_dir = os.path.join("scenario_origin", config.map, f"scenario_{scenario_id:02d}_scenarios")
    new_center, new_road_waypoints, selected_waypoints_idx = load_scenario(config, dist, waypoints_sparse, save_dir)
    if len(selected_waypoints_idx) > 0:
        center = new_center
        road_waypoints = new_road_waypoints

    def onclick(event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))
        if int(event.button) == 1:
            # left click to add or remove waypoints from the list
            waypoints_dist = np.linalg.norm(road_waypoints[:, :2] - [event.xdata, -event.ydata], axis=1)
            if waypoints_dist.min() < 5:
                idx = waypoints_dist.argmin()
                if idx in selected_waypoints_idx:
                    selected_waypoints_idx.remove(idx)
                else:
                    selected_waypoints_idx.append(idx)

            draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
            set_title(ax)
            plt.draw()

        elif int(event.button) == 3:
            # right click to save scenario
            if len(selected_waypoints_idx) < 1:
                set_title(ax, "Need at lease 1 waypoints to create a scenario.")
                plt.draw()
            else:
                # save routes to a numpy file
                selected_waypoints = np.take(road_waypoints, selected_waypoints_idx, axis=0)
                save_scenario(config, save_dir, selected_waypoints)

                # clear selected waypoints
                if config.scenario_idx < 0:
                    selected_waypoints_idx.clear()
                    draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
                    set_title(ax, "Scenario create success! Click to create more scenarios.")
                    plt.draw()
                else:
                    set_title(ax, "Scenario change success! You can keep edit the scenario.")
                    plt.draw()

    # visualize waypoints
    road_waypoints = select_waypoints(waypoints_sparse, center, dist)
    fig, ax = plt.subplots(figsize=(12, 12))
    draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
    set_title(ax)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Town_Safebench_Light')
    parser.add_argument('--save_dir', type=str, default="scenario_data/route_new_map")
    parser.add_argument('--scenario', type=int, default=5)
    parser.add_argument('--scenario_idx', type=int, default=-1)
    parser.add_argument('--road', type=str, default='auto', choices=['auto', 'intersection', 'straight'],
                        help='Create routes based on a intersection or a straight road.')
    parser.add_argument('--multi_rotation', action='store_true',
                        help='Create multiple symmetrical routes.'
                             'When creating routes that involve an intersection, the code will generate four routes, '
                             'each rotated 90 degrees around the center of the intersection. '
                             'When creating routes alone a straight road, the code will generate two routes, '
                             'each rotated 180 degrees around the center of the road. ')

    args = parser.parse_args()

    main(args)
