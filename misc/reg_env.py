from gym import envs
from gym.envs.registration import register
import highway_env

""" Outputs all the registered environments."""

all_envs = envs.registry.values()
env_ids = [env_spec.id for env_spec in all_envs]
ids = sorted(env_ids)
for id in ids:
    print(id)