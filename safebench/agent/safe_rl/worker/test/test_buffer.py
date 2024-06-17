import gym
from cpprb import ReplayBuffer
import numpy as np

env = gym.make("Pendulum-v0")

env_dict = {
    'act': {
        'dtype': np.float32,
        'shape': 2
    },
    'done': {
        'dtype': np.float32,
        'shape': 1
    },
    'obs': {
        'dtype': np.float32,
        'shape': 3
    },
    'obs2': {
        'dtype': np.float32,
        'shape': 3
    },
    'rew': {
        'dtype': np.float32,
        'shape': 1
    }
}
print(env_dict)

rb = ReplayBuffer(10, env_dict)

# obs = env.reset()

# for i in range(2000):
#     act = env.action_space.sample()
#     next_obs, rew, done, _ = env.step(act)

#     rb.add(obs=obs, act=act, next_obs=next_obs, rew=rew, done=done)
#     # Create `dict` for `ReplayBuffer.add`

#     if done:
#         obs = env.reset()
#         rb.on_episode_end()
#     else:
#         obs = next_obs
steps = 8
act = np.ones((steps, 2))
obs = np.ones((steps, 3)) * 5
obs2 = obs
rew = np.ones(steps) * 3
done = np.zeros(steps)
done[4] = 1

for i in range(steps):
    rb.add(act=act[i], obs=obs[i], obs2=obs2[i], rew=rew[i], done=done[i])

sample = rb.get_all_transitions()
rb.clear()
print(sample)

for i in range(steps):
    rb.add(act=act[i], obs=obs[i], obs2=obs2[i], rew=rew[i], done=done[i])
    rb.on_episode_end()

sample = rb.get_all_transitions()
rb.clear()
print(sample)