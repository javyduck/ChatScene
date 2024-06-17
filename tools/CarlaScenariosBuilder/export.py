import os.path
from copy import deepcopy
from export_routes import main as export_routes
from export_scenarios import main as export_scenarios


def main(config):
    export_routes(deepcopy(config))
    export_scenarios(deepcopy(config))


if __name__ == '__main__':
    import shutil
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Town_Safebench_Light')
    parser.add_argument('--save_dir', type=str, default="scenario_data/route_new_map")
    parser.add_argument('--scenario', type=int, default=-1)

    args = parser.parse_args()

    main(args)

    # TODO: remove them in the final version
    root = '/home/kinova/Documents/Carla/SafeBench_v2/SafeBench_v2-main/safebench/scenario'
    scenario_type_file = os.path.join(root, 'config/scenario_type_new_map/standard.json')
    assert os.path.isfile(scenario_type_file)
    shutil.copy2('scenario_data/route_new_map/standard.json', scenario_type_file)
    scenario_dir = os.path.join(root, 'scenario_data')
    shutil.rmtree(os.path.join(scenario_dir, 'route_new_map'))
    shutil.copytree('scenario_data', scenario_dir, dirs_exist_ok=True)


