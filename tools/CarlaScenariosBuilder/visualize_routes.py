import numpy as np
import os
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt

from utilities import parse_route, select_waypoints, get_view_centers, get_map_centers


def draw(ax, map_name, route_waypoints, centers, waypoints_sparse, zoom):
    if zoom:
        draw_zoom(ax, route_waypoints, centers, waypoints_sparse)
    else:
        draw_global(ax, map_name, route_waypoints, waypoints_sparse)


def draw_global(ax, map_name, route_waypoints, waypoints_sparse):
    ax.cla()
    ax.plot(waypoints_sparse[:, 0], -waypoints_sparse[:, 1], 'o', color='y')
    for waypoints in route_waypoints:
        ax.plot(waypoints[:, 0], -waypoints[:, 1], '-o', color='r')
        ax.plot(waypoints[0, 0], -waypoints[0, 1], 'o', color='g')
        ax.plot(waypoints[-1, 0], -waypoints[-1, 1], 'o', color='b')
        ax.text(waypoints[0, 0] + 8, -waypoints[0, 1] + 8, "Start", bbox=dict(facecolor='green', alpha=0.7))
        ax.text(waypoints[-1, 0] + 8, -waypoints[-1, 1] + 8, "End", bbox=dict(facecolor='red', alpha=0.7))

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


def draw_zoom(ax, route_waypoints, centers, waypoints_sparse):
    # get center of the waypoints
    geo_center = route_waypoints[0][:, :2].mean(0)
    if centers is None:
        center = geo_center
    else:
        center_dists = np.linalg.norm(np.array(centers) - geo_center, axis=1)
        center = centers[center_dists.argmin()]
    dist = 120

    ax.cla()
    road_waypoints = select_waypoints(waypoints_sparse, center, dist)
    ax.plot(waypoints_sparse[:, 0], -waypoints_sparse[:, 1], 'o', color='y')
    for road_waypoint in road_waypoints:
        length = 4
        x, y, z, pitch, yaw, roll = road_waypoint
        yaw = yaw / 180 * np.pi
        dx, dy = length * np.cos(yaw), length * np.sin(yaw)
        plt.arrow(x, -y, dx, -dy, color='y')

    for waypoints in route_waypoints:
        ax.plot(waypoints[:, 0], -waypoints[:, 1], '-o', color='r')
        ax.plot(waypoints[0, 0], -waypoints[0, 1], 'o', color='g')
        ax.plot(waypoints[-1, 0], -waypoints[-1, 1], 'o', color='b')
        ax.text(waypoints[0, 0] + 8, -waypoints[0, 1] + 8, "Start", bbox=dict(facecolor='green', alpha=0.7))
        ax.text(waypoints[-1, 0] + 8, -waypoints[-1, 1] + 8, "End", bbox=dict(facecolor='red', alpha=0.7))

    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def set_title(ax, map_name, route_file, title=None):
    if title is None:
        title = "Left and right click to change route.\nMiddle click to zoom."
    title = f"Visualizing route: {route_file} on map {map_name}.\n" + title
    ax.set_title(title, fontsize=12, loc='left')


def main(config):
    # get all routes of the scenarios
    scenario_id = config.scenario
    route_dir = os.path.join(config.save_dir, f"scenario_{scenario_id:02d}_routes")
    route_files = os.listdir(route_dir)
    route_files = list(filter(lambda x: x.lower().endswith(".xml"), route_files))
    route_files.sort()
    route_file_num = len(route_files)
    route_idx = 0
    route_file = os.path.join(route_dir, route_files[route_idx])
    route_waypoints, maps_names = parse_route(route_file)
    map_name = maps_names[0]

    # sparse waypoints used for visualization
    waypoints_sparse = np.load(f"map_waypoints/{map_name}/sparse.npy")

    # initialize potential center position
    centers = get_view_centers(map_name)
    zoom = False

    assert route_file_num > 0, "No route to visualize."

    def onclick(event):
        nonlocal route_idx, map_name, route_file, route_waypoints, waypoints_sparse, centers, zoom

        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))

        new_route_id = route_idx
        if int(event.button) == 1:
            # left click to visualize next route
            new_route_id = (route_idx + 1) % route_file_num
        elif int(event.button) == 3:
            # right click to visualize previous route
            new_route_id = (route_idx - 1) % route_file_num
        elif int(event.button) == 2:
            # middle click to zoom
            zoom = not zoom

        if new_route_id != route_idx:
            route_idx = new_route_id
            route_file = os.path.join(route_dir, route_files[route_idx])
            route_waypoints, map_names = parse_route(route_file)
            if map_name != map_names[0]:

                map_name = map_names[0]
                # sparse waypoints used for visualization
                waypoints_sparse = np.load(f"map_waypoints/{map_name}/sparse.npy")

                # initialize potential center position
                centers = get_view_centers(map_name)


        if 1 <= int(event.button) <= 3:
            draw(ax, map_name, route_waypoints, centers, waypoints_sparse, zoom)
            set_title(ax, map_name, route_file)
            plt.draw()

    # visualize waypoints
    fig, ax = plt.subplots(figsize=(12, 12))
    draw(ax, map_name, route_waypoints, centers, waypoints_sparse, zoom)
    set_title(ax, map_name, route_file)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Town_Safebench_Light')
    parser.add_argument('--save_dir', type=str, default="scenario_data/route_new_map")
    parser.add_argument('--scenario', type=int, required=True)

    args = parser.parse_args()

    main(args)
