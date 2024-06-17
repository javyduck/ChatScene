'''
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-06 00:21:23
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

# for planning scenario
from safebench.agent.dummy import DummyAgent
from safebench.agent.rl.sac import SAC
from safebench.agent.rl.ddpg import DDPG
from safebench.agent.rl.ppo import PPO
from safebench.agent.rl.td3 import TD3
from safebench.agent.basic import CarlaBasicAgent
from safebench.agent.behavior import CarlaBehaviorAgent
from safebench.agent.safe_rl.rl_agent import RLAgent

# for perception scenario
from safebench.agent.object_detection.yolov5 import YoloAgent
from safebench.agent.object_detection.faster_rcnn import FasterRCNNAgent

AGENT_POLICY_LIST = {
    'dummy': DummyAgent,
    'basic': CarlaBasicAgent,
    'behavior': CarlaBehaviorAgent,
    'yolo': YoloAgent,
    'sac': SAC,
    'ddpg': DDPG,
    'ppo': PPO,
    'safe_sac': RLAgent,
    'safe_ppo': RLAgent,
    'safe_td3': RLAgent,
    'safe_ddpg': RLAgent,
    'td3': TD3,
    'rl': RLAgent,
    'faster_rcnn': FasterRCNNAgent,
}
