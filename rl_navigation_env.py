import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneNavigationEnv(gym.Env):
    """
    PPO navigation environment.

    The drone learns goal-directed navigation in a 2D continuous environment.
    This represents the RL navigation layer used by the Gazebo controller.

    Observation:
    [drone_x, drone_y, target_x, target_y, dx, dy]

    Actions:
    0 = move north
    1 = move south
    2 = move west
    3 = move east
    4 = stay
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.world_limit = 6.0
        self.step_size = 0.35
        self.max_steps = 120

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=np.array([-6, -6, -6, -6, -12, -12], dtype=np.float32),
            high=np.array([6, 6, 6, 6, 12, 12], dtype=np.float32),
            dtype=np.float32,
        )

        # Rectangular building obstacle in the center
        self.building_min = np.array([-1.4, -1.2], dtype=np.float32)
        self.building_max = np.array([1.4, 1.2], dtype=np.float32)

        self.drone_pos = None
        self.target_pos = None
        self.steps = 0

    def _inside_building(self, pos):
        return (
            self.building_min[0] <= pos[0] <= self.building_max[0]
            and self.building_min[1] <= pos[1] <= self.building_max[1]
        )

    def _get_obs(self):
        delta = self.target_pos - self.drone_pos
        return np.array(
            [
                self.drone_pos[0],
                self.drone_pos[1],
                self.target_pos[0],
                self.target_pos[1],
                delta[0],
                delta[1],
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        start_positions = [
            np.array([-4.5, 4.5], dtype=np.float32),
            np.array([4.5, 4.5], dtype=np.float32),
            np.array([4.5, -4.5], dtype=np.float32),
            np.array([-4.5, -4.5], dtype=np.float32),
        ]

        target_positions = [
            np.array([0.0, -3.0], dtype=np.float32),
            np.array([-3.5, -2.5], dtype=np.float32),
            np.array([3.5, -2.5], dtype=np.float32),
            np.array([0.0, 3.5], dtype=np.float32),
        ]

        self.drone_pos = start_positions[self.np_random.integers(0, len(start_positions))].copy()
        self.target_pos = target_positions[self.np_random.integers(0, len(target_positions))].copy()

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        old_pos = self.drone_pos.copy()
        old_dist = np.linalg.norm(self.target_pos - self.drone_pos)

        move = np.array([0.0, 0.0], dtype=np.float32)

        if action == 0:
            move[1] = self.step_size
        elif action == 1:
            move[1] = -self.step_size
        elif action == 2:
            move[0] = -self.step_size
        elif action == 3:
            move[0] = self.step_size
        elif action == 4:
            move[:] = 0.0

        proposed_pos = self.drone_pos + move
        proposed_pos = np.clip(proposed_pos, -self.world_limit, self.world_limit)

        reward = -0.02
        terminated = False
        truncated = False

        # Collision penalty with building
        if self._inside_building(proposed_pos):
            reward -= 2.0
            proposed_pos = old_pos
        else:
            self.drone_pos = proposed_pos

        new_dist = np.linalg.norm(self.target_pos - self.drone_pos)

        # Reward moving closer to target
        reward += (old_dist - new_dist) * 1.5

        # Small penalty for staying still
        if action == 4:
            reward -= 0.1

        # Goal reward
        if new_dist < 0.45:
            reward += 8.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}
