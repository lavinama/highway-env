from gym import envs

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
ids = sorted(env_ids)
for id in ids:
    print(id)