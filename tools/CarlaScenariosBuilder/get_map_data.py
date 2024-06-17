import carla
import os
import argparse
import numpy as np

import create_scenarios


def generate_waypoints(world, dist):
    waypoint_list = world.get_map().generate_waypoints(dist)
    waypoints = []
    for item in waypoint_list:
        x = item.transform.location.x
        y = item.transform.location.y
        z = item.transform.location.z
        pitch = item.transform.rotation.pitch
        yaw = item.transform.rotation.yaw
        roll = item.transform.rotation.roll
        waypoints.append([x, y, z, pitch, yaw, roll])
    return np.asarray(waypoints)


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)

    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--map',
        default="Town_Safebench_Light",
        type=str,)

    args = argparser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = client.load_world(args.map)
    waypoints_sparse = generate_waypoints(world, 8.0)
    waypoints_dense = generate_waypoints(world, 1.0)
    save_dir = f'map_waypoints/{args.map}'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'sparse.npy'), waypoints_sparse)
    np.save(os.path.join(save_dir, 'dense.npy'), waypoints_dense)
    print("Waypoints output success.")


if __name__ == '__main__':
    main()

