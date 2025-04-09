import torch
import torch.nn as nn 
import gymnasium as gym
from replay_buffer import ReplayBuffer
from policy import Policy, EpsGreedy
import random 
import numpy as np
from shared import device

class Agent:
      def __init__(self, env, algorithm:nn.Module, policy:Policy, configuration_script:dict):
            self.env = env 
            self.script = configuration_script
            self.local_network = self.initialize(algorithm)
            self.target_network = self.initialize(algorithm)
            self.buffer = ReplayBuffer(self.script['buffer_size'])
            self.optimizer = torch.optim.Adam(self.local_network.parameters(), self.script['learning_rate'])
            self.criterion = nn.MSELoss()
            self.t = 0
            self.batch_size = self.script['batch_size']
            self.gamma = self.script['gamma']
            self.update_period = self.script['update_period']
            self.action_size = env.action_space.n
            self.policy = policy
            

            
      def initialize(self, algorithm):
            state_dim = self.env.observation_space.low.shape[0]
            n_actions = self.env.action_space.n
            if self.script['algorithm'] == 'dueling_dqn':
                  model = algorithm(state_dim, n_actions, True) if self.script['use_max'] else algorithm(state_dim, n_actions, False)

            model = self.xavier_init(model)
            return model

      def xavier_init(self, model):
            for p in model.parameters():
                  if len(p.shape) > 1:
                        nn.init.xavier_uniform_(p)
            return model


      def step(self, state, action, reward, next_state, done):

            self.buffer.append(state, action, reward, next_state, done) 
            if self.buffer.__len__() >= self.batch_size:
                  experiences = self.buffer.sample(self.batch_size)
                  self.learn(experiences)
            self.t += 1
            if self.t % self.update_period == 0:
                  self.target_network.load_state_dict(self.local_network.state_dict())
                  self.target_network.eval()

      def learn(self, experiences):
            self.local_network.train()
            self.optimizer.zero_grad()

            states, actions, rewards, next_states, dones = experiences
            next_actions = self.local_network(next_states).detach().argmax(1).unsqueeze(1)
            Q_target_next = self.target_network(next_states).gather(1, next_actions)
            Q_target = rewards + (self.gamma * Q_target_next * (1 - dones))
    
            Q_predicted = self.local_network(states).gather(1, actions)
    
            loss = self.criterion(Q_predicted, Q_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), 10)
            self.optimizer.step()

      def act(self, state, eps):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            return self.policy.get_action(self.local_network, state, eps)

            
            
      
            
            