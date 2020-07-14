from gym_foo.envs.game import Game,Square
from PIL import Image
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.game = Game()

    # all possible actions and environment data
    self.action_space = spaces.Discrete(4)        

    self.observation_space = spaces.Box(np.zeros(9), np.zeros(9), dtype=np.uint8)

  def step(self, action):
    # Execute one time step within the environment
    self.game.action(action)
    reward = self.game.evaluate()
    obs = self.game.observe()
    done = self.game.is_done()
    return obs, reward, done, {}

  def reset(self):
    # Reset the state of the environment to an initial state
    del self.game
    self.game = Game()
    obs = self.game.observe()
    return obs

  def render(self, mode='human'):
    # Render the environment to the screen
    self.game.view()

  def close(self):
    return self.game.end()
