import numpy as np
import gym
from gym import spaces
from stable_baselines.common.env_checker import check_env
from gym.envs.registration import register

from objs import Car, Environment
from renderer import Renderer

class CarNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CarNavigationEnv, self).__init__()

        # initializing environment and car objects
        self.env = Environment()
        self.car = Car()

        # defining the action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, 0] + [0] * (self.env.GRID_ROWS * self.env.GRID_COLS)),
            high=np.array([self.env.GRID_COLS, self.env.GRID_ROWS, np.pi, np.inf] + [1] * (self.env.GRID_ROWS * self.env.GRID_COLS)),
            dtype=np.float32
        )

        # setting limit
        self.max_steps = 1000
        self.current_step = 0

    def step(self, action):
        self.current_step += 1

        # perform action
        steer_angle = action[0]
        acceleration = action[1]

        x_shift = self.car.update_pos(steer_angle)
        self.car.accelerate(acceleration)
        self.env.shift(x_shift)

        # Check if done
        done = self.env.intersectsWith(self.car) or self.current_step >= self.max_steps

        # Calculate reward
        reward = 1 if not done else -100  # Positive reward for staying alive, large negative reward for collision

        # Get observation
        observation = self._get_obs()

        # additional info
        info = {}
        return observation, reward, done, info
    
    def reset(self):
        self.env = Environment()
        self.car = Car()
        return self._get_obs()
    
    def render(self, mode='human'):
        if self.renderer is None:
            self.renderer = Renderer(self.env, self.car)
        return self.renderer.render(mode)

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
    
    def _get_obs(self):
        return np.concatenate([
            self.car.pos,
            [self.car.angle, self.car.speed],
            self.env.grid.flatten()
        ]).astype(np.float32)
  

if __name__ == '__main__':
    register(
        id='CarNavigation-v0',
        entry_point='train2:CarNavigationEnv',
    )
