import numpy as np
import gym
from gym import spaces
from gym.envs.registration import register
from gym import error as gym_error
import time
import matplotlib
import matplotlib.pyplot as plt
import pickle

from objs import Car, Environment
from renderer import Renderer


MODEL_REGISTER_NAME = 'CarNavigation-v0'

MIN_ANGLE_TO_PUNISH = 45    # at and above this angle, a small negative reward is added
EPSILON = np.cos(MIN_ANGLE_TO_PUNISH * np.pi / 180)

ACTION_TO_STEER_ANGLE_MULTIPLIER = 90
UNNECESSARY_STEER_ANGLE_LIMIT = 0.1


mapInteractionToReward = {
    "None": 0,
    "Collided": -5,
    "Hit Border": -7,
    "Gap Crossed": 15,
    "Not Moving Forward": -1,
    "Moving Forward": 1,
    "Steering": -0.5
}

def RewardSystem(done, x_shift, envInteractionType, steer_angle):
    reward = 0
    if done:
        return mapInteractionToReward[envInteractionType]   # collision (with obstacle or border)
    
    if abs(steer_angle) > UNNECESSARY_STEER_ANGLE_LIMIT:
        reward += mapInteractionToReward['Steering']        # Avoid unnecessary steering
    
    if x_shift <= EPSILON:
        reward += mapInteractionToReward['Not Moving Forward']     # not moving forward enough
    else:
        reward += mapInteractionToReward['Moving Forward']         # moving forward
    
    reward += mapInteractionToReward[envInteractionType]    # passes through a gap
    return reward


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
        self.max_steps = 10000
        self.current_step = 0
        
        # rendering speed (during testing)
        self.fps = self.metadata["render.fps"]

        # keeping episode count and path
        self.episode_count = 0
        self.all_episode_paths = []

        # Store car positions for plotting
        self.positions = []

    def step(self, action):
        self.current_step += 1

        # perform action
        steer_angle = action[0] * ACTION_TO_STEER_ANGLE_MULTIPLIER
        acceleration = action[1]

        x_shift = self.car.update_pos(steer_angle)
        self.car.accelerate(acceleration)
        self.env.shift(x_shift)

        prev_x_pos = 0 if self.positions == [] else self.positions[-1][0]
        self.positions.append([prev_x_pos + x_shift, self.car.pos[1]])  # Store the position
        
        # Check if done
        isIntersecting, envInteractionType = self.env.intersectsWith(self.car)
        done = isIntersecting or self.current_step >= self.max_steps

        # Calculate reward
        reward = RewardSystem(done, x_shift, envInteractionType, steer_angle)
        
        # Get observation
        observation = self._get_obs()

        # additional info
        info = {}
        
        return observation, reward, done, info
    
    def reset(self):
        if self.current_step > 0:  # Not the first reset
            self.all_episode_paths.append(np.array(self.positions))
            self.episode_count += 1

        self.env = Environment()
        self.car = Car()
        self.current_step = 0
        self.positions = []  # Reset positions for new episode
        
        if self.renderer is not None:
            self.renderer.env = self.env
            self.renderer.car = self.car
        
        return self._get_obs()
    
    def render(self, mode='human'):
        if mode == 'human' and self.renderer is None:
            self.renderer = Renderer(self.env, self.car)
        if self.renderer is not None:
            self.renderer.render(mode)
            time.sleep(1.0 / self.fps)  # Control the FPS

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        
        # Close the plot
        plt.close('all')
    
    def _get_obs(self):
        return {
            'angle': np.array([self.car.angle], dtype=np.float32),
            'position': np.array(self.car.pos, dtype=np.float32),
            'speed': np.array([self.car.speed], dtype=np.float32),
            'grid': self.env.grid.astype(np.int32)
        }
    
    def display_all_paths(self, savePlotData = False):
        # write the data into a file (optional)
        if savePlotData:
            with open("episode_paths.pkl", 'wb') as fpaths:
                pickle.dump(self.all_episode_paths, fpaths)
        
        plt.figure(figsize=(10, 8))
        for i, path in enumerate(self.all_episode_paths):
            plt.plot(path[:, 0], path[:, 1], label=f'Episode {i+1}')
        
        #plt.xlim(0, self.env.GRID_COLS)
        plt.ylim(0, self.env.GRID_ROWS)
        plt.title(f"Paths for {self.episode_count} Episodes")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        
        plt.show(block=True)  # This will keep the plot window open


def register_env():
    try:
        register(
            id=MODEL_REGISTER_NAME,
            entry_point='CarNavigationGymEnv:CarNavigationEnv'
        )
        print("Model registration successful.")
    except gym_error.Error:
        print("Model may be already registered.")