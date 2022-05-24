from typing import Dict, Tuple

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.envs.common.graphics import StaticEnvViewer


class GridlockEnv(AbstractEnv):

    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    NUM_ROADS = 4
    ROAD_LENGTH = 30  # [m]
    DISTANCE_BETWEEN_VEHICLES = 2
    DISTANCE_BETWEEN_ROADS = 10

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h",
                                 "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False
                }
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
            "duration": 13,  # [s]
            "destination": "o1",
            "controlled_vehicles": 8,
            "initial_vehicle_count": 0,
            "spawn_probability": 0,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
            "scaling": 5.5 * 1.3,
            "collision_reward": -100,
            "high_speed_reward": 1,
            "arrived_reward": 50,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": False
        })
        return config

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.viewer = StaticEnvViewer(self, offset=np.array([0, 0]))

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = self.config["collision_reward"] * vehicle.crashed \
                 + self.config["high_speed_reward"] * vehicle.speed

        reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else reward
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        reward = 0 if not vehicle.on_road else reward
        return reward

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

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, done, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 2 for right-turns
            - 1 for straight lanes
            - 0 for left-turns

        The code for nodes in the road network is:
        (o:outer | (i:inner | c:central) + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        turn_radius = lane_width  # [m}
        outer_distance = turn_radius + lane_width / 2 + self.DISTANCE_BETWEEN_ROADS / 2
        inner_distance = -turn_radius + lane_width / 2 + self.DISTANCE_BETWEEN_ROADS / 2
        access_length = self.ROAD_LENGTH  # [m]
        lateral_offset = lane_width / 2 + self.DISTANCE_BETWEEN_ROADS / 2
        max_priority = 2

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(self.NUM_ROADS):
            angle = np.radians(90 * corner)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lateral_offset, access_length + outer_distance])
            end = rotation @ np.array([lateral_offset, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[c, c], priority=max_priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % self.NUM_ROADS),
                         CircularLane(r_center, turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=max_priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([inner_distance, -inner_distance]))
            net.add_lane("cr" + str((corner - 1) % self.NUM_ROADS), "cl" + str((corner - 2) % self.NUM_ROADS),
                         CircularLane(l_center, turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[c, s], priority=max_priority - 2, speed_limit=10))

            # Straight
            start = rotation @ np.array([lateral_offset, outer_distance])
            mid1 = rotation @ np.array([lateral_offset, inner_distance])
            mid2 = rotation @ np.array([lateral_offset, -inner_distance])
            end = rotation @ np.array([lateral_offset, -outer_distance])
            net.add_lane("ir" + str(corner), "cl" + str((corner - 1) % self.NUM_ROADS),
                         StraightLane(start, mid1, line_types=[n, s], priority=max_priority - 1, speed_limit=10))
            net.add_lane("cl" + str((corner - 1) % self.NUM_ROADS), "cr" + str((corner - 1) % self.NUM_ROADS),
                         StraightLane(mid1, mid2, line_types=[c, c], priority=max_priority - 1, speed_limit=10))
            net.add_lane("cr" + str((corner - 1) % self.NUM_ROADS), "il" + str((corner + 2) % self.NUM_ROADS),
                         StraightLane(mid2, end, line_types=[n, s], priority=max_priority - 1, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lateral_offset, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lateral_offset, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % self.NUM_ROADS), "o" + str((corner - 1) % self.NUM_ROADS),
                         StraightLane(end, start, line_types=[c, c], priority=max_priority, speed_limit=10))

            road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
            self.road = road

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
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

        # Challenger vehicle
        # self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, speed_deviation=0)

        # Controlled vehicles
        self.controlled_vehicles = []
        offsets = np.zeros(self.NUM_ROADS)
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                ("o{}".format(ego_id % self.NUM_ROADS),
                 "ir{}".format(ego_id % self.NUM_ROADS),
                 0)
            )
            destination = self.config["destination"]\
                          or "o" + str((ego_id + self.np_random.randint(0, 2)) % self.NUM_ROADS)
            offsets[ego_id % self.NUM_ROADS] += self.np_random.rand(1)
            ego_position = ego_lane.position(
                self.ROAD_LENGTH + self.DISTANCE_BETWEEN_ROADS + 2 - offsets[ego_id % self.NUM_ROADS],
                ((self.np_random.rand(1) * 2) - 1))
            ego_heading = ego_lane.heading + \
                          ((self.np_random.rand(1) * 2) - 1)[0] * np.pi / 12
            ego_vehicle = self.action_type.vehicle_class(self.road, ego_position, ego_heading, 0)
            offsets[ego_id % self.NUM_ROADS] += ego_vehicle.LENGTH + self.DISTANCE_BETWEEN_VEHICLES
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            """
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)
            """

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.rand() > spawn_probability:
            return

        route = self.np_random.choice(range(self.NUM_ROADS), size=2, replace=False)
        route[1] = (route[0] + 2) % self.NUM_ROADS if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
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
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - self.NUM_ROADS * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


    register(
        id='gridlock-v0',
        entry_point='highway_env.envs:GridlockEnv',
    )