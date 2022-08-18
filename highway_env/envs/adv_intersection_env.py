from typing import Dict, Tuple

from gym.envs.registration import register
import numpy as np
import math

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad, CheckRegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle


class AdvIntersectionEnv(AbstractEnv):

    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    END_ROAD_OFFSET = 5
    NUM_ROADS = 4
    ROAD_LENGTH = 100  # [m]
    MIN_DIST_VEHICLES = 2
    MAX_DIST_VEHICLES = 10
    DISTANCE_BETWEEN_ROADS = 5
    # Added Attributes
    ZERO_SUM_REWARDS = False
    FAILMAKER_ADVRL = False
    CHECK_REG_ROAD = True

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
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
            "hash_intersection": False,
            "controlled_vehicles": 1,
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
            "scaling": 5.5 * 1.3,
            "scaling_factor": 1,
            "collision_reward": -100,
            "high_speed_reward": 1,
            "speed_to_reward": 3.5,
            "arrived_reward": 30,
            "rule_break_reward": -50,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": False
        })
        return config

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        total_reward = sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)
        # print("Total reward:", total_reward)
        return total_reward

    def calc_adv_reward(self, vehicle: Vehicle) -> float:
        """Calculate the adversarial reward = 
        the current contribution of NPC / the total previous contributions"""
        for vehicle in self.controlled_vehicles:
            if vehicle.ego:
                ego_vehicle = vehicle
                break
        # Calculate total contribution
        total_contr = 0
        for position in vehicle.prev_positions:
            prev_dist = np.linalg.norm(position - ego_vehicle.position)
            total_contr += math.exp(-2*prev_dist)
        # Calculate current contribution
        dist = np.linalg.norm(vehicle.position - ego_vehicle.position)
        contr = math.exp(-2*dist)
        adv_reward = contr/total_contr
        return adv_reward

    def calc_rule_break(self, vehicle: Vehicle) -> float:
        if self.road.rule_broken:
            adv_reward = self.config["rule_break_reward"]
        else:
            adv_reward = 0
        return adv_reward
        
    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        # Add the previous positions of the vehicle
        vehicle.prev_positions.append(vehicle.position)

        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = self.config["collision_reward"] * vehicle.crashed \
                 + self.config["high_speed_reward"] * (vehicle.speed - self.config["speed_to_reward"])

        reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else reward
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        reward = 0 if not vehicle.on_road else reward
        if self.ZERO_SUM_REWARDS:
            # r_npc = - r_ego
            if vehicle.ego is False:
                print("NPC reward: ", -reward)
                return -reward
            print("Ego rewards: ", reward)
        if self.FAILMAKER_ADVRL:
            # reward function of FailMaker_AdvRL
            if vehicle.ego is False:
                pers_reward = reward
                adv_reward = self.calc_adv_reward(vehicle)
                reward = pers_reward + self.config["scaling_factor"] * adv_reward
        if self.CHECK_REG_ROAD:
            # reward function encourages to break rules of the road
            if vehicle.ego is False:
                pers_reward = reward
                adv_reward = self.calc_rule_break(vehicle)
                reward = - pers_reward + self.config["scaling_factor"] * adv_reward
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
        """Return a dictionary of additional information"""
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        info["agents_crashed"] = tuple(vehicle.crashed for vehicle in self.controlled_vehicles)
        info["agents_arrived"] = tuple(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
        info["agent_names"] = tuple(vehicle.name for vehicle in self.controlled_vehicles)
        return info

    def _reset(self) -> None:
        if not self.config.get("hash_intersection", False):
            self._make_road()
        else:
            self._make_hash_road()

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
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left] | c:center) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius - lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = self.ROAD_LENGTH  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
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
            l_center = rotation @ (np.array([-outer_distance + lane_width * 2, outer_distance - lane_width * 2]))
            # l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("c" + str((corner - 1) % self.NUM_ROADS), "c" + str((corner + 2) % self.NUM_ROADS),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            """
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % self.NUM_ROADS),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            """
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            mid = rotation @ np.array([lane_width / 2, 0])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "c" + str((corner - 1) % self.NUM_ROADS),
                         StraightLane(start, mid, line_types=[s, n], priority=priority, speed_limit=10))
            net.add_lane("c" + str((corner - 1) % self.NUM_ROADS), "il" + str((corner + 2) % self.NUM_ROADS),
                         StraightLane(mid, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % self.NUM_ROADS), "o" + str((corner - 1) % self.NUM_ROADS),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        if self.CHECK_REG_ROAD:
            road = CheckRegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        else:
            road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_hash_road(self) -> None:
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
                                      clockwise=False, line_types=[n, n], priority=max_priority - 2, speed_limit=10))

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
            offsets[ego_id % self.NUM_ROADS] += self.np_random.rand(1) * (self.MAX_DIST_VEHICLES - self.MIN_DIST_VEHICLES)
            ego_position = ego_lane.position(self.ROAD_LENGTH - self.END_ROAD_OFFSET - offsets[ego_id % self.NUM_ROADS],
                                             ((self.np_random.rand(1) * 2) - 1))
            ego_heading = ego_lane.heading + \
                          ((self.np_random.rand(1) * 2) - 1)[0] * np.pi / 12
            ego_vehicle = self.action_type.vehicle_class(self.road, ego_position, ego_heading, 0)
            offsets[ego_id % self.NUM_ROADS] += ego_vehicle.LENGTH + self.MIN_DIST_VEHICLES
            if ego_lane.on_lane(ego_position):
                # Car can also go right
                destination = self.config["destination"] \
                              or "o" + str((ego_id + self.np_random.randint(1, 4)) % 4)
            else:
                # Car can only move forwards or left
                destination = self.config["destination"] \
                              or "o" + str((ego_id + self.np_random.randint(1, 3)) % 4)
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass
            
            # Classify the ego_vehicle as ego or npc
            if ego_id == 0:
                ego_vehicle.ego = True
                ego_vehicle.name = "ego"
            else:
                ego_vehicle.ego = False
                ego_vehicle.name = "npc_" + str(ego_id)

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
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


class MultiAgentAdvIntersectionEnv(AdvIntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                 "type": "MultiAgentAction",
                 "action_config": {
                     "type": "DiscreteMetaAction",
                     "lateral": False,
                     "longitudinal": True,
                     "target_speeds": [-3, 0, 4.5, 9]
                 }
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 15
                }
            },
            "controlled_vehicles": 2
        })
        return config

class HashAdvIntersectionEnv(MultiAgentAdvIntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "hash_intersection": True
        })
        return config

class MultiAgentDeadlockIntersectionEnv(MultiAgentAdvIntersectionEnv):
    END_ROAD_OFFSET = -7.5
    MIN_DIST_VEHICLES = 2
    MAX_DIST_VEHICLES = 3

class HashDeadlockAdvIntersectionEnv(HashAdvIntersectionEnv):
    END_ROAD_OFFSET = -7.5
    MIN_DIST_VEHICLES = 2
    MAX_DIST_VEHICLES = 3

class ContinuousAdvIntersectionEnv(AdvIntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
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
        })
        return config


TupleMultiAgentIntersectionEnv = MultiAgentWrapper(MultiAgentAdvIntersectionEnv)


register(
    id='intersection-adv-v0',
    entry_point='highway_env.envs:AdvIntersectionEnv',
)

register(
    id='intersection-adv-v1',
    entry_point='highway_env.envs:ContinuousAdvIntersectionEnv',
)

register(
    id='intersection-multi-agent-adv-v0',
    entry_point='highway_env.envs:MultiAgentAdvIntersectionEnv',
)

register(
    id='intersection-multi-agent-adv-v1',
    entry_point='highway_env.envs:TupleMultiAgentAdvIntersectionEnv',
)

register(
    id='intersection-multi-agent-deadlock-adv-v0',
    entry_point='highway_env.envs:MultiAgentDeadlockAdvIntersectionEnv',
)

register(
    id='hash-intersection-adv-v0',
    entry_point='highway_env.envs:HashAdvIntersectionEnv',
)

register(
    id='hash-deadlock-adv-v0',
    entry_point='highway_env.envs:HashDeadlockAdvIntersectionEnv',
)
