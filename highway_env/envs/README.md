*Note: Only the highway and intersection environments allow for multi-agent settings*
### Default environment configuration `common/abstract.py`
```
{
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
    "manual_control": False,
    "real_time_rendering": False
}
```
### Intersection environment configuration `intersection_env.py`
##### "intersection-v0"
```
{
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "absolute": True,
        "flatten": False,
        "observe_intentions": False
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": False,
        "target_speeds": [0, 4.5, 9]
    },
    "duration": 13,  # [s]
    "destination": "o1",
    "controlled_vehicles": 1,
    "initial_vehicle_count": 10,
    "spawn_probability": 0.6,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.6],
    "scaling": 5.5 * 1.3,
    "collision_reward": -5,
    "high_speed_reward": 1,
    "arrived_reward": 1,
    "reward_speed_range": [7.0, 9.0],
    "normalize_reward": False,
    "offroad_terminal": False
}
```
##### "intersection-v1"
```
{
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off", "ang_off"],
    },
    "action": {
        "type": "ContinuousAction",
        "steering_range": [-np.pi / 3, np.pi / 3],
        "longitudinal": True,
        "lateral": True,
        "dynamical": True
    },
}
```
##### "intersection-multi-agent-v0" 
```
{
    "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                "lateral": False,
                "longitudinal": True
            }
    },
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics"
        }
    },
    "controlled_vehicles": 2
}
```
##### "intersection-multi-agent-v1"
```
# Same as "intersection-multi-agent-v0" 
```
### Merge environment configuration `merge_env.py`
```
# The default config plus
{
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.2,
    "merging_speed_reward": -0.5,
    "lane_change_reward": -0.05,
}
```
