import numpy as np 
import torch 
torch.manual_seed(0)

class Policy:
      def __init__(self, *args) -> None:
            pass
      def get_action(self, q_network, state):
            pass 


class EpsGreedy(Policy):
      def __init__(self,action_space_size:int):
            self.action_space_size = action_space_size
      def get_action(self, q_network, state, eps):
            q_network.eval()
            with torch.no_grad():
                  action_values = q_network(state)
            q_network.train()
            if np.random.rand() > eps:
                  return np.argmax(action_values.cpu().numpy())
            else:
                  return np.random.choice(np.arange(self.action_space_size))

