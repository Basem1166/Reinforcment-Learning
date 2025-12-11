import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torchvision import models

# Initialize weights for better training stability
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# ============= CarRacing Image Encoder =============

class CarRacingEncoder(nn.Module):
    """Simple CNN encoder for CarRacing-v3 (96x96x3 RGB).

    Outputs a feature vector that can be fed into existing MLP policies/Q-nets.
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        # Input: (C=3, H=96, W=96)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> 32 x 23 x 23
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> 64 x 10 x 10
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> 64 x 8 x 8
            nn.ReLU(inplace=True),
        )

        conv_out_dim = 64 * 8 * 8
        self.fc = nn.Linear(conv_out_dim, feature_dim)

        self.apply(weights_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input formats from FrameStackObservation
        # FrameStackObservation outputs: (num_frames, H, W) for single state
        # Or (B, num_frames, H, W) for batch
        
        if x.dim() == 3:  # Single state: (num_frames, H, W)
            x = x.unsqueeze(0)  # -> (1, num_frames, H, W)
        elif x.dim() == 4:
            # Check if we have (B, H, W, C) format instead of (B, C, H, W)
            # If third dimension is much larger than second, it's likely (B, H, W, C)
            if x.shape[1] > 4 and x.shape[3] <= 4:  # Likely (B, H, W, num_frames)
                x = x.permute(0, 3, 1, 2)  # -> (B, num_frames, H, W)

        x = x.float() / 255.0
        x = self.conv(x)
        # Use reshape instead of view to handle non-contiguous tensors
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class PretrainedResNetEncoder(nn.Module):
    """Pretrained ResNet18 encoder for CarRacing-v3 with frame averaging.
    
    Uses ImageNet-pretrained ResNet18 for better feature extraction.
    Handles grayscale 96x96 input with 4-frame stacking.
    """
    
    def __init__(self, feature_dim: int = 256, freeze_backbone: bool = False):
        super().__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # ResNet18 outputs 512D features
        self.fc = nn.Linear(512, feature_dim)
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.apply(weights_init_)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (num_frames, H, W) from FrameStackObservation
               or (B, num_frames, H, W) from batched input
        
        Returns:
            Features of shape (B, feature_dim) averaged across frames
        """
        # Handle single input case
        if x.dim() == 3:  # (num_frames, H, W)
            x = x.unsqueeze(0)  # -> (1, num_frames, H, W)
        
        batch_size = x.size(0)
        num_frames = x.size(1)
        
        # Normalize to [0, 1]
        x = x.float() / 255.0
        
        # Reshape to (batch_size * num_frames, H, W)
        x = x.reshape(-1, x.size(2), x.size(3))
        
        # Convert grayscale to RGB by replicating channels
        x = x.unsqueeze(1)  # -> (B*num_frames, 1, H, W)
        x = x.repeat(1, 3, 1, 1)  # -> (B*num_frames, 3, H, W)
        
        # Resize to 224x224 for ResNet18
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features via ResNet18
        x = self.backbone(x)  # -> (B*num_frames, 512)
        
        # Project to feature_dim
        x = self.fc(x)  # -> (B*num_frames, feature_dim)
        
        # Reshape back to (B, num_frames, feature_dim) and average across frames
        x = x.reshape(batch_size, num_frames, -1)
        x = x.mean(dim=1)  # -> (B, feature_dim)
        
        return x


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
