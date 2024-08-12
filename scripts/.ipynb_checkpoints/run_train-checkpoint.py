''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-22 18:45:01
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''
import setGPU
import traceback
import os
import os.path as osp

import torch 
from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner import CarlaRunner

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--output_dir', type=str, default='log')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))

    parser.add_argument('--max_episode_step', type=int, default=300)
    parser.add_argument('--auto_ego', action='store_true')
    parser.add_argument('--mode', '-m', type=str, default='eval', choices=['train_scenario', 'train_agent', 'eval'])
    parser.add_argument('--agent_cfg', nargs='*', type=str, default=['adv_scenic.yaml'])
    parser.add_argument('--scenario_cfg', nargs='*', type=str, default=['train_scenario_scenic.yaml'])
    parser.add_argument('--continue_agent_training', '-cat', type=bool, default=False)
    parser.add_argument('--continue_scenario_training', '-cst', type=bool, default=False)

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')   

    parser.add_argument('--num_scenario', '-ns', type=int, default=1, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2002, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8002, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    parser.add_argument('--scenario_id', type=int, default=None)
    args = parser.parse_args()
    args_dict = vars(args)

    err_list = []
    for agent_cfg in args.agent_cfg:
        for scenario_cfg in args.scenario_cfg:
            # set global parameters
            set_torch_variable(args.device)
            torch.set_num_threads(args.threads)
            set_seed(args.seed)

            # load agent config
            agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', agent_cfg)
            agent_config = load_config(agent_config_path)

            # load scenario config
            scenario_config_path = osp.join(args.ROOT_DIR, 'safebench/scenario/config', scenario_cfg)
            scenario_config = load_config(scenario_config_path)
            
            ## modification
            if args.scenario_id:
                scenario_config['scenario_id'] = args.scenario_id
                
            agent_config['load_dir'] = osp.join(agent_config['load_dir'], f"scenario_{scenario_config['scenario_id']}")
            # Check if the directory exists; if not, create it
            if not osp.exists(agent_config['load_dir']):
                os.makedirs(agent_config['load_dir'])        
                
            # main entry with a selected mode
            agent_config.update(args_dict)
            args_dict['output_dir'] = osp.join('log', 'adv_train', args.mode, agent_config['policy_name'], f"{agent_cfg.split('.')[0]}", f"scenario_{scenario_config['scenario_id']}")
            scenario_config.update(args_dict)
            if scenario_config['policy_type'] == 'scenic':
                from safebench.scenic_runner import ScenicRunner
                scenario_config['num_scenario'] = 1 ### 'the num_scenario can only be one for scenic'
                runner = ScenicRunner(agent_config, scenario_config)
            else:
                runner = CarlaRunner(agent_config, scenario_config)

            # start running
            runner.run()

    for err in err_list:
        print(err[0], err[1], 'failed!')
        print(err[2])