import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Initialize weights for better training stability
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# ============= SAC Models =============

class SACQNetwork(nn.Module):
    """Q-Network for SAC: Q(s, a) -> scalar"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(SACQNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x = F.relu(self.fc1(xu))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SACGaussianPolicy(nn.Module):
    """Stochastic Gaussian policy for SAC"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, action_scale=1.0, log_std_min=-20, log_std_max=2):
        super(SACGaussianPolicy, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Action scale as buffer
        if isinstance(action_scale, (float, int)):
            action_scale_tensor = torch.tensor([action_scale] * action_dim, dtype=torch.float32)
        else:
            action_scale_tensor = torch.tensor(action_scale, dtype=torch.float32)
        self.register_buffer('action_scale', action_scale_tensor)
        
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        
        # Log probability with change of variables
        eps = 1e-6
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(self.action_scale + eps).sum(dim=-1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale
        
        return action, log_prob, mean


# ============= PPO Models =============

class PPOActor(nn.Module):
    """Actor network for PPO - outputs mean and log_std for Gaussian policy"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.apply(weights_init_)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        
        # Expand log_std to match batch size
        std = self.log_std.exp().expand_as(mean)
        
        return mean, std
    
    def get_dist(self, state):
        mean, std = self.forward(state)
        return Normal(mean, std)


class PPOCritic(nn.Module):
    """Critic network for PPO - estimates V(s)"""
    def __init__(self, obs_dim, hidden_dim=256):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        value = self.value(x)
        return value


# ============= TD3 Models =============

class TD3Actor(nn.Module):
    """Deterministic policy for TD3"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(TD3Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        
        self.apply(weights_init_)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action


class TD3Critic(nn.Module):
    """Twin Q-networks for TD3"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(TD3Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc4 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        # Q1
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        q1 = self.fc3(x1)
        
        # Q2
        x2 = F.relu(self.fc4(xu))
        x2 = F.relu(self.fc5(x2))
        q2 = self.fc6(x2)
        
        return q1, q2
    
    def Q1(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        q1 = self.fc3(x1)
        return q1
