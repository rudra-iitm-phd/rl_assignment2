import numpy as np 
from collections import deque, namedtuple
import random
import torch
from shared import device

class ReplayBuffer:
      def __init__(self, capacity:int):
            self.buffer = deque(maxlen = capacity)
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        

      def append(self, state, action, reward, next_state, done):
            e = self.experience(state, action, reward, next_state, done)
            self.buffer.append(e)

      def sample(self, batch_size):
            experiences = random.sample(self.buffer, k=batch_size)
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

            return (states, actions, rewards, next_states, dones)

      def __len__(self):
            return len(self.buffer)