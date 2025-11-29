import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# 1. Universal Actor (Policy)
# Handles both Discrete (Softmax) and Continuous (Mean+Std)
class UniversalActor(nn.Module):
    def __init__(self, obs_dim, action_space):
        super(UniversalActor, self).__init__()
        
        self.is_continuous = False
        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
            self.action_dim = action_space.shape[0]
        else:
            self.action_dim = action_space.n

        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Output Heads
        if self.is_continuous:
            self.mean_layer = nn.Linear(128, self.action_dim)
            self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))
        else:
            self.action_layer = nn.Linear(128, self.action_dim)
            
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.is_continuous:
            mean = self.mean_layer(x)
            std = torch.exp(self.log_std.expand_as(mean))
            return mean, std
        else:
            # Discrete: Return Probabilities
            return F.softmax(self.action_layer(x), dim=-1)

    def get_distribution(self, state):
        if self.is_continuous:
            mean, std = self.forward(state)
            return torch.distributions.Normal(mean, std)
        else:
            probs = self.forward(state)
            return torch.distributions.Categorical(probs)

# 2. Universal Critic (Value)
# Always outputs a scalar value V(s)
class UniversalCritic(nn.Module):
    def __init__(self, obs_dim):
        super(UniversalCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)
        
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value_layer(x)