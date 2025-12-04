import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

# Initialize weights for better training stability
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    """Q-Network for continuous actions: Q(s, a) -> scalar"""
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


class DiscreteQNetwork(nn.Module):
    """Q-Network for discrete actions: Q(s) -> Q-values for all actions"""
    def __init__(self, obs_dim, action_dim):
        super(DiscreteQNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Returns Q-values for all actions

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale=1.0):
        super(GaussianPolicy, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Two heads: one for Mean, one for Log Std Dev
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)
        
        # Keep action_scale as a tensor buffer for stable math
        if isinstance(action_scale, (float, int)):
            action_scale_tensor = torch.tensor([action_scale] * action_dim, dtype=torch.float32)
        else:
            action_scale_tensor = torch.as_tensor(action_scale, dtype=torch.float32)
        self.register_buffer("action_scale", action_scale_tensor)
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
        
        # Guard: replace any NaNs in std/mean before creating distribution
        if torch.isnan(mean).any() or torch.isnan(std).any():
            # Return zeros and very low log_prob to avoid crashing
            zeros_action = torch.zeros_like(mean)
            very_low_log_prob = torch.full((mean.shape[0], 1), -1e6, device=mean.device)
            return zeros_action, very_low_log_prob, mean

        normal = Normal(mean, std)
        
        # Reparameterization Trick: u = mean + std * noise
        x_t = normal.rsample()
        
        # Squash to [-1, 1]
        y_t = torch.tanh(x_t)
        
        # Scale to environment bounds
        action = y_t * self.action_scale
        
        # Numerically stable log-prob with change of variables
        eps = 1e-6
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # Account for scaling by action_scale (Jacobian term)
        log_prob = log_prob - torch.log(self.action_scale + eps).sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean


class CategoricalPolicy(nn.Module):
    """Categorical policy for discrete action spaces (SAC-Discrete)"""
    def __init__(self, obs_dim, action_dim):
        super(CategoricalPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.logits = nn.Linear(512, action_dim)
        
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)
        return logits

    def get_probs(self, state):
        """Returns action probabilities"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs

    def sample(self, state):
        """
        Sample action and compute log_prob.
        Returns: action (int tensor), log_prob, probs
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        # Numerical stability: clamp probs
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        
        dist = Categorical(probs)
        action = dist.sample()
        
        # Log prob of sampled action
        log_prob = dist.log_prob(action).unsqueeze(-1)
        
        return action, log_prob, probs

    def evaluate(self, state):
        """
        For policy update: returns probs and log_probs for ALL actions.
        Used in the SAC-Discrete policy loss.
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        # Clamp for numerical stability
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        log_probs = torch.log(probs)
        
        return probs, log_probs