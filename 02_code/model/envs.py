import os
import gym
import numpy as np
from matplotlib import pyplot as plt


class Trainer(gym.Env):
    """ Pokemon Battle Environment """
    def __init__(self, available, budget, max_i):
        super(Trainer, self).__init__()
        """ Initializing class """
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=0, high=max_i, shape=(6, ), dtype=np.float32)
        self.i = None
        self.max_i = max_i
        self.available = available
        self.budget = None
        self.battle_results = None
        self.picks = None
        self.done = None

    def _fight(self):
        pass

    def _reward(self):
        """ Reward value """
        if self.budget > 3500:
            reward = -100
        
        reward = np.mean(self.battle_results)
        
        return reward

    def _state(self):
        """ Get current state """
        pass


    def reset(self):
        """ Resets environment """
        pass

    def step(self, action):
        """ Action """
        pass

    def render(self):
        """ Plot of current state """
        pass