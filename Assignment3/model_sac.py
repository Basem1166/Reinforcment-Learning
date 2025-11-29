import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Initialize weights for better training stability
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        # Q-Network input: State + Action concatenated
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x = F.relu(self.fc1(xu))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale=1.0):
        super(GaussianPolicy, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Two heads: one for Mean, one for Log Std Dev
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)
        
        self.action_scale = action_scale
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        
        # Clamp log_std to maintain numerical stability 
        # (Standard practice in SAC implementations)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        # Reparameterization Trick (Step 14 in slide): a = mean + std * noise
        x_t = normal.rsample() 
        
        # Squash action to [-1, 1]
        y_t = torch.tanh(x_t)
        
        # Scale to environment bounds (e.g., if env action is [-2, 2])
        action = y_t * self.action_scale
        
        # Enforce Log Probability bounds (Correction for Tanh Squashing)
        log_prob = normal.log_prob(x_t)
        # The formula: log_pi(a|s) = log_mu(u|s) - sum(log(1 - tanh(u)^2))
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean