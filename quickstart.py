import time

import gym
import highway_env
import numpy as np
from matplotlib import pyplot as plt

NUM_CONTROLLED_VEHICLES = 2

env = gym.make('highwayadv-v0')
# env = gym.make('gridlock-v0')
# env = gym.make('intersection-multi-agent-v0')
# env = gym.make('intersection-multi-agent-deadlock-v0')
# env = gym.make('hash-deadlock-v0')
env.seed(0)

env.configure({"controlled_vehicles": NUM_CONTROLLED_VEHICLES}) # Two controlled vehicles
env.configure({"vehicles_count": 0})  # A single other vehicle, for the sake of visualisation
# env.configure({"initial_vehicle_count": 0})
# env.configure({"spawn_probability": 0})
# env.configure({"destination": "o1"})

env.config["lanes_count"] = 2

env.reset()
# actions = tuple(np.array([-0.1, 0])
#                 for agent in range(NUM_CONTROLLED_VEHICLES))
actions = tuple(0 for agent in range(NUM_CONTROLLED_VEHICLES))
for i in range(00):
    obs, reward, done, info = env.step(actions)
    # obs, reward, done, info = env.step(np.array([0.01, 0.1]))
    # obs, reward, done, info = env.step(0)
    env.render()

actions = tuple(2 for agent in range(NUM_CONTROLLED_VEHICLES))
for i in range(00):
    obs, reward, done, info = env.step(actions)
    # obs, reward, done, info = env.step(np.array([0.01, 0.1]))
    # obs, reward, done, info = env.step(0)

    env.render()

plt.imshow(env.render(mode="rgb_array"))
plt.title("Controlled vehicles are in yellow")
plt.show()
