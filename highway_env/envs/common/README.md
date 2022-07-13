### Types of Observations:
* "GrayscaleObservation": An observation class that collects directly what the simulator renders.
    ```python
    "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84), # (screen_width, screen_height)
            "stack_size": 4, # number of input channels N (N x W x H)
            "weights": [0.2989, 0.5870, 0.1140]  # weights for RGB conversion
        }
    ```
* "TimeToCollision": Observe the time to collision
* "Kinematics": Observe the kinematics of nearby vehicles.
    ```python
    "observation": {
            "env": "AbstractEnv", # The environment to observe
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"], # Names of features used in the observation
            "vehicles_count": 15, # Number of observed vehicles
            "absolute": True, # Use absolute coordinates
            "order": "sorted", # Order of observed vehicles. Values: sorted, shuffled
            "normalize": True, # Should the observation be normalized
            "clip": True, # Should the value be clipped in the desired range
            "see_behind": False, # Should the observation contains the vehicles behind
            "observe_intentions": False, # Observe the destinations of other vehicles
        }
    ```
* "OccupancyGrid": Observe an occupancy grid of nearby vehicles
* "KinematicsGoal"
* "AttributesObservation"
* "MultiAgentObservation"
    ```python
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics"
        }
    }
    ```
* "TupleObservation"
* "LidarObservation"
* "ExitObservation": Specific to exit_env, observe the distance to the next exit lane as part of a KinematicObservation.

### Types of actions
* "ContinuousAction": A continuous action space for throttle and/or steering angle.
    ```python
    "action": {
            "type": "ContinuousAction",
            "env": "AbstractEnv", # the environment
            "acceleration_range": (-5, 5.0), # the range of acceleration values [m/s²]
            "steering_range": [-np.pi / 3, np.pi / 3], # the range of steering values [rad]
            "speed_range": (0, 20), # the range of reachable speeds [m/s]
            "longitudinal": True, # enable throttle control
            "lateral": True, # enable steering control
            "dynamical": True, # whether to simulate dynamics (i.e. friction) rather than kinematics
            "clip": True # clip action to the defined range
        }
    ```
* "DiscreteAction": A discrete action space
    ```python
    "action": {
            "type": "DiscreteAction",
            "env": "AbstractEnv", # the environment
            "acceleration_range": (-5, 5.0), # the range of acceleration values [m/s²]
            "steering_range": [-np.pi / 3, np.pi / 3], # the range of steering values [rad]
            "longitudinal": True, # enable throttle control
            "lateral": True, # enable steering control
            "dynamical": True, # whether to simulate dynamics (i.e. friction) rather than kinematics
            "clip": True # clip action to the defined range
            "actions_per_axis", 3
        }
    ```
* "DiscreteMetaAction": A discrete action space of meta-actions: lane changes, and cruise control set-point.
    ```python
    "action": {
            "type": "DiscreteAction",
            "env": "AbstractEnv", # the environment
            "longitudinal": True, # enable throttle control
            "lateral": True, # enable steering control
            "target_speeds": , # the list of speeds the vehicle is able to track
        }
    ```
* "MultiAgentAction"
    ```python
    "action": {
            "type": "MultiAgentAction",
            "env": "AbstractEnv", # the environment
            "action_config": {
                "type": "DiscreteMetaAction",
                "lateral": False,
                "longitudinal": True
            }
        }
    ```