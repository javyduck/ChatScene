<!--
 * @Date: 2022-11-09 11:22:05
 * @LastEditTime: 2023-03-06 00:21:04
 * @Description: 
-->

# Modules and Functions

To help developers quickly understand the structure of the code, we briefly introduce important modules of Safebench.

## Carla Runner (safebench/carla_runner.py)

This is the entry point that manages all modules by hosting a loop to run all scenarios.

## Gym module (safebench/gym_carla/)

This is the gym-style interface for Carla. The implementation of environments is in the `envs` folder. Each env will contains one scenario and a vectorized wrapper is implemented in `env_wrapper.py` to manager all scenarios that simultaneously run on the same map.

## Agent Module (safebench/agent/)

The implementations of autonomous vehicle agents are placed here. 
The configuration files corresponding to these agents are placed in the `config` folder.
The saved model files corresponding to these agents are placed in the `model_ckpt` folder.

## Scenario Module (safebench/scenario/)

The implementations of traffic scenarios are placed here. 
The folder `scenario_data` stores data and model that scenarios use, including scenario routes, scenario models, and adversarial attack templates.
The folder `config` stores configurations of scenarios, where the `.yaml` files will call different types of scenarios in the `scenario_type` folder.
File `scenario_data_loader.py` contains a data loader to sample scenario configurations for training and evaluation.

The folder `srunner` contains partial files of Carla Scenario Runner for parsering configuration files (`scenario_configs`), scenario implementation (`scenarios`), and managing scenarios (`scenario_manager`). 
New scenarios implemented by the users should be placed into folder `scenarios`.