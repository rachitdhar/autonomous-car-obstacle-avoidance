import numpy as np
import gym
from gym import spaces
from gym.envs.registration import register
from gym import error as gym_error
import time

from objs import Car, Environment
from renderer import Renderer


MODEL_REGISTER_NAME = 'CarNavigation-v0'
EPSILON = 0.5

ACTION_TO_STEER_ANGLE_MULTIPLIER = 60.0

def timeBasedReward(timestep):
    return (np.sqrt(timestep) if timestep < 100 else 10)

class CarNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human'], "render.fps": 8}

    def __init__(self, render_mode=None):
        super(CarNavigationEnv, self).__init__()

        # initializing environment and car objects
        self.env = Environment()
        self.car = Car()
        self.renderer = None

        assert render_mode is None or render_mode in self.metadata["render.modes"]
        self.render_mode = render_mode

        # defining the action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'angle': spaces.Box(low=np.array([-np.pi / 2]), high=np.array([np.pi / 2]), dtype=np.float32),
            'position': spaces.Box(low=np.array([0, 0]), high=np.array([self.env.GRID_COLS, self.env.GRID_ROWS]), dtype=np.float32),
            'speed': spaces.Box(low=np.array([0]), high=np.array([100]), dtype=np.float32),
            'grid': spaces.Box(low=0, high=1, shape=(self.env.GRID_ROWS, self.env.GRID_COLS), dtype=np.int32)
        })

        # setting limit
        self.max_steps = 1000
        self.current_step = 0
        
        # rendering speed (during testing)
        self.fps = self.metadata["render.fps"]

        # to log the steer angle and x_shift, and see when collision occurs
        self.log_line = []

    def step(self, action):
        self.current_step += 1

        # perform action
        steer_angle = action[0] * ACTION_TO_STEER_ANGLE_MULTIPLIER
        acceleration = action[1]

        x_shift = self.car.update_pos(steer_angle)
        #self.car.accelerate(acceleration)
        self.env.shift(x_shift)

        self.log_line.append(f"{steer_angle} {x_shift}\n")
        
        # Check if done
        isIntersecting, gapPassingReward = self.env.intersectsWith(self.car)
        done = isIntersecting or self.current_step >= self.max_steps

        # Calculate reward
        reward = 1

        if done:
            reward = -100
            self.log_line.append(f"---COLLISION---")
        elif abs(x_shift) <= EPSILON:
            reward = -10
        else:
            reward += timeBasedReward(self.current_step)
            
        reward += gapPassingReward
        
        # Get observation
        observation = self._get_obs()

        # additional info
        info = {}
        
        return observation, reward, done, info
    
    def reset(self):
        self.env = Environment()
        self.car = Car()
        self.current_step = 0
        
        if self.renderer is not None:
            self.renderer.env = self.env
            self.renderer.car = self.car
        
        self.log_line.append("\n-------------\n")
        return self._get_obs()
    
    def render(self, mode='human'):
        if mode == 'human' and self.renderer is None:
            self.renderer = Renderer(self.env, self.car)
        if self.renderer is not None:
            self.renderer.render(mode)
            time.sleep(1.0 / self.fps)  # Control the FPS

    def close(self):
        with open('log.txt', 'a') as log:
            log.writelines(self.log_line)
        
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
    
    def _get_obs(self):
        return {
            'angle': np.array([self.car.angle], dtype=np.float32),
            'position': np.array(self.car.pos, dtype=np.float32),
            'speed': np.array([self.car.speed], dtype=np.float32),
            'grid': self.env.grid.astype(np.int32)
        }


def register_env():
    try:
        register(
            id=MODEL_REGISTER_NAME,
            entry_point='CarNavigationGymEnv:CarNavigationEnv'
        )
        print("Model registration successful.")
    except gym_error.Error:
        print("Model may be already registered.")