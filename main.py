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
# from trainer import train, mc_train


args = parser.parse_args()

configuration_script = args.__dict__

c = Configure()
shared.configuration_script = configuration_script
env, algorithm, rl_agent, policy, train = c.configure(configuration_script)

policy = policy(env.action_space.n) if policy is not None else None
rl_agent = rl_agent(env, algorithm, policy, configuration_script)

train(env, rl_agent)






