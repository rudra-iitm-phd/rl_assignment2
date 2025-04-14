import gymnasium as gym 
import numpy as np 
from algorithms import DuelingDQN, MonteCarloREINFORCE
from agent import DuelingDQNAgent, MonteCarloREINFORCEAgent
from policy import EpsGreedy
from trainer import Trainer


class Configure:
      def __init__(self):
            self.env = {
                  'cartpole':gym.make('CartPole-v1'),
                  'acrobot':gym.make('Acrobot-v1')
            }

            self.algo = {
                  'dueling_dqn':DuelingDQN,
                  'mc_reinforce':MonteCarloREINFORCE
            }

            self.agents = {
                  'dueling_dqn':DuelingDQNAgent,
                  'mc_reinforce':MonteCarloREINFORCEAgent
            }

            self.trainers = {
                  'dueling_dqn':Trainer('dueling_dqn').get_trainer(),
                  'mc_reinforce':Trainer('mc_reinforce').get_trainer()
            }

      def configure(self, script:dict):
            environment = self.env[script['env'].lower()]
            algorithm = self.algo[script['algorithm'].lower()]
            rl_agent = self.agents[script['algorithm'].lower()]
            
            policy = EpsGreedy if script['algorithm'].lower() == 'dueling_dqn' else None
            trainer_function = self.trainers[script['algorithm'].lower()]
      
            return environment, algorithm, rl_agent, policy, trainer_function