"""
Main script where the file runs
"""
import gym
import highway_env
import pprint
import matplotlib.pyplot as plt

env = gym.make("highwayadv-v0")
env.config["lanes_count"] = 2
env.reset()

for _ in range(3):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    env.render()

plt.imshow(env.render(mode="rgb_array"))
plt.show()
