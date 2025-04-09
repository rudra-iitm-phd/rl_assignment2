import gymnasium as gym
import numpy as np 
from configure import Configure
from argument_parser import parser
from agent import Agent
from policy import EpsGreedy
from collections import deque
from shared import device
import torch
from trainer import train


args = parser.parse_args()

configuration_script = args.__dict__

c = Configure()

env, algorithm = c.configure(configuration_script)
print(env.action_space.n)

# print(f'Action space : {env.action_space}\nObservation space : {env.observation_space}\nReward range : {env.reward_range}')

print(env.observation_space.low.shape[0])

policy = EpsGreedy(env.action_space.n)
rl_agent = Agent(env, algorithm, policy, configuration_script)


# def train(agent, n_episodes=10000, max_t=1000, eps_start=1, eps_end=0.01, eps_decay=0.985):

#     scores_window = deque(maxlen=100)
#     ''' last 100 scores for checking if the avg is more than 195 '''

#     eps = eps_start
#     ''' initialize epsilon '''

#     score_history = []

#     for i_episode in range(1, n_episodes+1):
#         state, _ = env.reset()
#         score = 0
#         for t in range(max_t):
#             action = agent.act(state, eps)
#             next_state, reward, done, _ , _ = env.step(action)
#             agent.step(state, action, reward, next_state, done)
#             state = next_state
#             score += reward
#             if done:
#                 break

#         scores_window.append(score)

#         eps = max(eps_end, eps_decay*eps)
#         ''' decrease epsilon '''

#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

#         if i_episode % 100 == 0:
#            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#            score_history.append([i_episode, np.mean(scores_window)])
#         if np.mean(scores_window)>=195.0:
#            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#            score_history.append([i_episode, np.mean(scores_window)])
#            break
#     return True, score_history, i_episode

train(env, rl_agent)






