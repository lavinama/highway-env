from typing import Tuple

from abc import abstractmethod
from gym import Env
from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs import Action, GoalEnv
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.graphics import StaticEnvViewer
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.lane import StraightLane, LineType, AbstractLane, \
    CircularLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark


def _reached_goal(vehicle):
    return vehicle.goal._is_colliding(vehicle, 0)[0]


class DeadlockEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    NUM_ROADS = 4
    ROAD_LENGTH = 40
    DISTANCE_BETWEEN_VEHICLES = 2

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        # self.viewer = StaticEnvViewer(self)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "KinematicsGoal",
                    "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": False,
                    "longitudinal": True,
                    "target_speeds": [-3, 0, 4.5, 9]
                }
            },
            "controlled_vehicles": 4,
            "collision_reward": -100,
            "high_speed_reward": 1,
            "speed_to_reward": 3.5,
            "arrived_reward": 50,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
            "offroad_terminal": False,
            "scaling": 7,
        })
        return config

    def _is_terminal(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
               or (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
               or self.has_arrived(vehicle)

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        return info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        self._clear_vehicles()
        return obs, reward, done, info

    def _clear_vehicles(self) -> None:
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles
                              and not _reached_goal(vehicle)]

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner
        + [r:right, l:left])
        + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = self.ROAD_LENGTH  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        self.goals = []
        for corner in range(self.NUM_ROADS):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % self.NUM_ROADS),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % self.NUM_ROADS),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % self.NUM_ROADS),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            end_lane = StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10)
            net.add_lane("il" + str((corner - 1) % self.NUM_ROADS), "o" + str((corner - 1) % self.NUM_ROADS), end_lane)

            # Goal
            longitudinal = end_lane.length
            self.goals.append(Landmark(self.road,
                                end_lane.position(longitudinal, 0),
                                heading=end_lane.heading))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

        # Append Goals to road
        for goal in self.goals:
            self.road.objects.append(goal)

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        offsets = np.zeros(self.NUM_ROADS)
        for ego_id in range(self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                ("o{}".format(ego_id % self.NUM_ROADS),
                 "ir{}".format(ego_id % self.NUM_ROADS),
                 0)
            )
            offsets[ego_id % self.NUM_ROADS] += self.np_random.rand(1)
            ego_position = ego_lane.position(self.ROAD_LENGTH + 7.5 - offsets[ego_id % self.NUM_ROADS],
                                             ((self.np_random.rand(1) * 2) - 1))
            ego_heading = ego_lane.heading + \
                          ((self.np_random.rand(1) * 2) - 1)[0] * np.pi / 12
            vehicle = self.action_type.vehicle_class(self.road, ego_position, ego_heading, 0)
            offsets[ego_id % self.NUM_ROADS] += vehicle.LENGTH + self.DISTANCE_BETWEEN_VEHICLES

            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            # Allocate one goal to each vehicle
            vehicle.goal = self.goals[self.np_random.randint(self.NUM_ROADS)]

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = self.config["collision_reward"] * vehicle.crashed \
                 + self.config["high_speed_reward"] * (vehicle.speed - self.config["speed_to_reward"])

        reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else reward
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        reward = 0 if not vehicle.on_road else reward
        return reward

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]


class DeadlockEnv8(DeadlockEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "controlled_vehicles": 8
        })
        return config


register(
    id='deadlock-v0',
    entry_point='highway_env.envs:DeadlockEnv',
)

register(
    id='deadlock-v1',
    entry_point='highway_env.envs:DeadlockEnv8',
)
