import random
import torch
import torch.nn as nn
from collections import deque, namedtuple

# A named tuple to store transitions
# This needs to be imported by agent.py
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class QNetwork(nn.Module):
    """
    A simple MLP Q-Network.
    """
    def __init__(self, state_dim, action_dim, layer1_size, layer2_size):
        """
        Initializes the network.
        
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            layer1_size (int): Number of neurons in the first hidden layer.
            layer2_size (int): Number of neurons in the second hidden layer.
        """
        super(QNetwork, self).__init__()
        
        # Use the arguments to define the network structure
        self.network = nn.Sequential(
            nn.Linear(state_dim, layer1_size),
            nn.ReLU(),
            nn.Linear(layer1_size, layer2_size),
            nn.ReLU(),
            nn.Linear(layer2_size, action_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): The input state.
        
        Returns:
            torch.Tensor: The Q-values for each action.
        """
        # PyTorch's loss functions expect float32
        if x.dtype != torch.float32:
             x = x.float()
        return self.network(x)


class ReplayMemory:
    """
    A finite-sized replay buffer.
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly samples a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)