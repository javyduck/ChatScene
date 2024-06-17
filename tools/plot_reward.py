''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-01 14:29:59
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


with open('./results.pkl', 'rb') as f:
    data = pkl.load(f)

episode = data['episode']
reward = data['episode_reward']

plt.plot(episode, reward)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.grid()
plt.xlim([0, 100])
plt.tight_layout()
plt.savefig('reward.png', dpi=300)
