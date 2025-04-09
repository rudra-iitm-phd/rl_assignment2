import torch 
import torch.nn as nn 
import torch.nn.functional as F

class DuelingDQN(nn.Module):
      def __init__(self, state_dim:int, n_actions:int, use_max:bool):
            super(DuelingDQN, self).__init__()

            self.use_max = use_max

            self.feature_embedding = nn.Sequential(
                  nn.Linear(state_dim, 256),
                  nn.ReLU(),
                  nn.Linear(256, 128),
                  nn.ReLU()
            )

            self.value = nn.Sequential(
                  nn.Linear(128, 64),
                  nn.ReLU(),
                  nn.Linear(64, 1)
            )

            self.advantage = nn.Sequential(
                  nn.Linear(128, 64),
                  nn.ReLU(),
                  nn.Linear(64, n_actions)
            )
            

      def forward(self, x):
            x = self.feature_embedding(x)
            V = self.value(x)
            A = self.advantage(x)
            if self.use_max:
                  return V + (A - A.max(dim = 1, keepdim = True).values)
            
            return V + (A - A.mean(dim = 1, keepdim = True))