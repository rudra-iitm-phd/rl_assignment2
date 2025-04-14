import gymnasium as gym
import numpy as np 
from configure import Configure
from argument_parser import parser
from agent import DuelingDQNAgent, MonteCarloREINFORCEAgent
from policy import EpsGreedy
from collections import deque
from shared import device
import torch
import shared
import wandb
import sweep_configuration
wandb.login(key = "f7cc061a6cf1c6d4f2791a84e81d1d16ee8adc8b")
np.random.seed(0)


def create_name(configuration:dict):
      l = [f'{k}-{v}' for k,v in configuration.items() if k not in ['wandb_entity', 'wandb_project', 'wandb_sweep', 'sweep_id']]
      return '_'.join(l)


def train():
      with wandb.init(entity = config['wandb_entity'],project = config['wandb_project'], name = create_name(config), config = config):

            sweep_config = wandb.config

            config.update(sweep_config)

            print(config)

            env, algorithm, rl_agent, policy, train_fn = c.configure(config)

            env.observation_space.seed(0)
            env.action_space.seed(0)

            policy = policy(env.action_space.n) if policy is not None else None
            rl_agent = rl_agent(env, algorithm, policy, config)

            train_fn(env, rl_agent, wandb)


      pass



if __name__ == '__main__':

      args = parser.parse_args()

      config = args.__dict__

      c = Configure()
      shared.configuration_script = config
      if args.wandb_sweep:
            sweep_config = sweep_configuration.sweep_config
            if not args.sweep_id :
                  sweep_id = wandb.sweep(sweep_config, project=config['wandb_project'], entity=config['wandb_entity'])
            else:
                  sweep_id = args.sweep_id
            
            wandb.agent(sweep_id, function=train, count=10)
            
            wandb.finish()
      else:
            train()
      






