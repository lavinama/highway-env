"""
Main script where the file runs
"""

import gym
import highway_env
from matplotlib import pyplot as plt

import pprint

env = gym.make("highway-v0")
pprint.pprint(env.config)