''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 17:18:58
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='safebench.gym_carla.envs:CarlaEnv',
)
