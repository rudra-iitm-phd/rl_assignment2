import gymnasium as gym 
import numpy as np 
from algorithms import DuelingDQN


class Configure:
      def __init__(self):
            self.env = {
                  'cartpole':gym.make('CartPole-v1'),
                  'acrobot':gym.make('Acrobot-v1')
            }

            self.algo = {
                  'dueling_dqn':DuelingDQN
            }

      def configure(self, script:dict):
            environment = self.env[script['env'].lower()]
            algorithm = self.algo[script['algorithm'].lower()]
            return environment, algorithm