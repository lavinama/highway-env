from typing import Dict, Tuple

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs import GoalEnv, observation_factory
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, \
    AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Landmark, Obstacle, VariatingObstacle


class DeadIntersectionEnv(AbstractEnv, GoalEnv):
    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "KinematicsGoal",
                    "vehicles_count": 4,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h",
                                 "sin_h"],
                    "goal_features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "order": "shuffled",
                    # TODO: consider observing intentions
                    "observe_intentions": False
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "ContinuousAction",
                    "lateral": True,
                    "longitudinal": True
                }
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "duration": 13,  # [s]
            "destinations": ["o1", "o2", "o3", "o0"],
            "controlled_vehicles": 4,
            "initial_vehicle_count": 0,
            "spawn_probability": 0,
            "screen_width": 550,  # TODO: make smaller (i.e. 500 after dev)
            "screen_height": 550,
            "centering_position": [0.5, 0.55],
            "scaling": 5.5 * 1.3,
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "high_speed_reward": 1,
            "arrived_reward": 1,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": False
        })
        return config

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        obs = self.observation_type.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        return sum(self.compute_reward(agent_obs['achieved_goal'],
                                       agent_obs['desired_goal'], {})
                   for agent_obs in obs)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        # TODO: get a better idea what this would be,
        # Whether it would be based on the vehicle and, if so,
        # what it could replace
        raise NotImplementedError

    # TODO: deprecate this. Replace everywhere with goal-based reward
    def _agent_reward_no_goal(self, action: int, vehicle: Vehicle) -> float:
        scaled_speed = utils.lmap(self.vehicle.speed,
                                  self.config["reward_speed_range"], [0, 1])
        reward = self.config["collision_reward"] * vehicle.crashed \
                 + self.config["high_speed_reward"] * np.clip(scaled_speed, 0,
                                                              1)

        reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else reward
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"],
                                         self.config["arrived_reward"]], [0, 1])
        reward = 0 if not vehicle.on_road else reward
        return reward

    def compute_reward(self, achieved_goal: np.ndarray,
                       desired_goal: np.ndarray, info: dict,
                       p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal),
                                np.array(self.config["reward_weights"])), p)

    def _is_terminal(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or all(
            self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config[
                   "policy_frequency"] \
               or (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config[
                   "policy_frequency"] \
               or self.has_arrived(vehicle)

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward_no_goal(action, vehicle) for vehicle in
            self.controlled_vehicles)
        info["agents_dones"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in
            self.controlled_vehicles)
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_goals()
        self._make_obstacles()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def _make_obstacles(self):
        width = self.road.network.lanes_list()[0].width_at(0)
        length = self.road.network.lanes_list()[0].length
        offset = 1.5
        base = 5
        for j in [-1, 1]:
            for k in [-1, 1]:
                obstacle = VariatingObstacle(road=self.road,
                                             position=[j * (length - base), k * (length - base)],
                                             length=30, width=30)
                self.road.objects.append(obstacle)


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, done, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 25  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c],
                                      priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius,
                                      angle + np.radians(180),
                                      angle + np.radians(270),
                                      line_types=[n, c], priority=priority,
                                      speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2,
                                             left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius,
                                      angle + np.radians(0),
                                      angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n],
                                      priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n],
                                      priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4),
                         "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c],
                                      priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random,
                             record_history=self.config["show_trajectories"])
        self.road = road

    def _make_goals(self):
        """
        Goals are synonymous to the destinations where agents can arrive to in
        the intersection, only they are required for agents trained on a
        Continuous or Discrete ActionSpace to reach the goal (since they don't
        use a Controller to keep the lane), as well as for the reward function.

        One destination will be assigned to each ego_agent in _make_vehicles().
        There will be exactly one goal for each end of lane (so exactly 4 in a
        basic cross-intersection).
        """
        self.goals = dict()
        for corner in range(4):
            lane = self.road.network.get_lane(
                (f"il{corner}", f"o{corner}", None))
            self.goals[f"o{corner}"] = Landmark(self.road,
                                                lane.position(lane.length, 0),
                                                heading=lane.heading_at(60))
            self.road.objects.append(self.goals[f"o{corner}"])

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(),
              self.road.step(1 / self.config["simulation_frequency"])) for _ in
             range(self.config["simulation_frequency"])]

        # Controlled vehicles
        self.controlled_vehicles = []
        self.goal_of = dict()
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                (f"o{ego_id % 4}", f"ir{ego_id % 4}", 0))
            destination = self.config["destinations"][ego_id] \
                          or f"o{self.np_random.randint(1, 4)}"
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(25 + 8, 0),
                speed=0,
                heading=ego_lane.heading_at(60))
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                    ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            self.goal_of[ego_vehicle] = self.goals[destination]
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(
                        v.position - ego_vehicle.position) < 0.1:
                    self.road.vehicles.remove(v)
            # TODO: play with distances in collision logic

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.rand() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, (
            "o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                            speed=8 + self.np_random.randn() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        out_of_bounds = lambda vehicle: (abs(vehicle.lane.local_coordinates(
                                            vehicle.position)[1]) >= vehicle.lane.length
                                         or \
                                         abs(vehicle.lane.local_coordinates(
                                             vehicle.position)[0]) >= vehicle.lane.length)

        self.road.vehicles = [vehicle for vehicle in self.road.vehicles
                              if not (out_of_bounds(vehicle) or self.has_arrived(vehicle))]

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def has_arrived(self, vehicle: Vehicle) -> bool:
        """The episode is over if the ego vehicles reached their goals."""
        return self.goal_of[vehicle].hit

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


register(
    id='dead-intersection-v0',
    entry_point='highway_env.envs:DeadIntersectionEnv',
)
