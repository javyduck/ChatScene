import numpy as np
import os
import matplotlib.pyplot as plt
from utilities import build_route, select_waypoints, get_nearist_waypoints, rotate_waypoints, get_map_centers


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
        ax.plot(waypoints[:, 0], -waypoints[:, 1], '-o', color='r')
        ax.plot(waypoints[0, 0], -waypoints[0, 1], 'o', color='g')
        ax.text(waypoints[0, 0] + 8, -waypoints[0, 1] + 8, "Start", bbox=dict(facecolor='green', alpha=0.7))

        if len(selected_waypoints_idx) > 1:
            ax.plot(waypoints[-1, 0], -waypoints[-1, 1], 'o', color='b')
            ax.text(waypoints[-1, 0] + 8, -waypoints[-1, 1] + 8, "End", bbox=dict(facecolor='red', alpha=0.7))

    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def set_title(ax, title=None):
    if title is None:
        title = "Left click:  select or remove waypoints.\nRight click:  save route to file."
    ax.set_title(title, fontsize=25, loc='left')


def get_route_id(config, save_dir):
    scenario_id = config.scenario
    # get the index of the current route
    route_id = 0
    while os.path.isfile(os.path.join(save_dir, f'scenario_{scenario_id:02d}_route_{route_id:02d}.xml')):
        route_id += 1
    return route_id


def create_route_intersection(config, selected_waypoints):
    # rotate waypoints
    # selected_waypoints = np.take(road_waypoints, selected_waypoints_idx, axis=0)
    local_waypoints = []
    if config.multi_rotation:
        for i in range(4):
            theta = i * np.pi / 2
            rotated_waypoints = rotate_waypoints(selected_waypoints, [0, 0], theta)
            local_waypoints.append(rotated_waypoints)
    else:
        local_waypoints.append(selected_waypoints)
    local_waypoints = np.vstack(local_waypoints)

    # rotate waypoints around the map center
    map_waypoints = []
    for i in range(4):
        theta = i * np.pi / 2
        rotated_waypoints = rotate_waypoints(local_waypoints, [100, 100], theta)
        map_waypoints.append(rotated_waypoints)
    map_waypoints = np.vstack(map_waypoints)
    return map_waypoints


def create_route_straight(config, selected_waypoints):
    # rotate waypoints
    local_waypoints = []
    if config.multi_rotation:
        for i in range(2):
            theta = i * np.pi
            rotated_waypoints = rotate_waypoints(selected_waypoints, [100, 0], theta)
            local_waypoints.append(rotated_waypoints)
    else:
        local_waypoints.append(selected_waypoints)
    local_waypoints = np.vstack(local_waypoints)

    # rotate waypoints around the map center
    map_waypoints = []
    for i in range(4):
        theta = i * np.pi / 2
        rotated_waypoints = rotate_waypoints(local_waypoints, [100, 100], theta)
        map_waypoints.append(rotated_waypoints)
    map_waypoints = np.vstack(map_waypoints)
    return map_waypoints


def create_route(config, selected_waypoints, waypoints_dense):
    # get point shift in x and y locations
    if config.road == 'intersection':
        all_routes_waypoints = create_route_intersection(config, selected_waypoints)
    elif config.road == 'straight':
        all_routes_waypoints = create_route_straight(config, selected_waypoints)
    else:
        raise ValueError("--road must be 'intersection' or 'straight'.")

    # save waypoints
    scenario_id = config.scenario
    save_dir = os.path.join(config.save_dir, f"scenario_{scenario_id:02d}_routes")
    os.makedirs(save_dir, exist_ok=True)

    route_length = len(selected_waypoints)
    route_num = len(all_routes_waypoints) // route_length
    for idx in range(route_num):
        route_waypoints = all_routes_waypoints[route_length * idx: route_length * (idx + 1)]
        real_route_waypoints = []

        # find the closest waypoints from the dense waypoints
        for route_waypoint in route_waypoints:
            waypoints_dist = np.linalg.norm(waypoints_dense[:, :2] - route_waypoint[:2].reshape((1, -1)), axis=1)
            idx = waypoints_dist.argmin()
            real_route_waypoint = waypoints_dense[idx]
            real_route_waypoints.append(real_route_waypoint)
            if waypoints_dist.min() > 2:
                print(f"waypoint {route_waypoint} can not be found on the map, "
                      f"assigned to the nearist waypoint {real_route_waypoint}")

        # save waypoints to file
        route_id = get_route_id(config, save_dir)
        save_file = os.path.join(save_dir, f'scenario_{scenario_id:02d}_route_{route_id:02d}.xml')
        build_route(real_route_waypoints, route_id, config.map, save_file)


def save_waypoints(config, save_dir, selected_waypoints):
    route_id = config.route
    if config.route < 0:
        # create new route
        route_id = 0
        while os.path.isfile(os.path.join(save_dir, f'route_{route_id:02d}.npy')):
            route_id += 1

    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'route_{route_id:02d}.npy')
    np.save(save_file, selected_waypoints)


def load_route(config, dist, waypoints_sparse, save_dir):
    # load route if route is set
    center = None
    road_waypoints = None
    selected_waypoints_idx = []
    if config.route >= 0:
        # check if route is created
        route_file = os.path.join(save_dir, f'route_{config.route:02d}.npy')
        if os.path.isfile(route_file):
            # load existing routes
            selected_waypoints = np.load(route_file)

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
    save_dir = os.path.join("scenario_origin", config.map, f"scenario_{scenario_id:02d}_routes")
    new_center, new_road_waypoints, selected_waypoints_idx = load_route(config, dist, waypoints_sparse, save_dir)
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
            # right click to save waypoints
            if len(selected_waypoints_idx) < 2:
                set_title(ax, "Need at lease 2 waypoints to create a route.")
                plt.draw()
            else:
                # save routes to a numpy file
                selected_waypoints = np.take(road_waypoints, selected_waypoints_idx, axis=0)
                save_waypoints(config, save_dir, selected_waypoints)

                # clear selected waypoints
                if config.route < 0:
                    selected_waypoints_idx.clear()
                    draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
                    set_title(ax, "Route create success! Click to create more routes.")
                    plt.draw()
                else:
                    set_title(ax, "Route change success! Click to keep change the route.")
                    plt.draw()

    # visualize waypoints

    fig, ax = plt.subplots(figsize=(12, 12))
    draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
    set_title(ax)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Town_Safebench_Light')
    # parser.add_argument('--save_dir', type=str, default="scenario_data/route_new_map")
    parser.add_argument('--scenario', type=int, required=True)
    parser.add_argument('--route', type=int, default=-1)
    parser.add_argument('--road', type=str, default='auto', choices=['auto', 'intersection', 'straight'],
                        help='Create routes based on a intersection or a straight road.')
    # parser.add_argument('--multi_rotation', action='store_true',
    #                     help='Create multiple symmetrical routes.'
    #                          'When creating routes that involve an intersection, the code will generate four routes, '
    #                          'each rotated 90 degrees around the center of the intersection. '
    #                          'When creating routes alone a straight road, the code will generate two routes, '
    #                          'each rotated 180 degrees around the center of the road. ')

    args = parser.parse_args()

    main(args)
