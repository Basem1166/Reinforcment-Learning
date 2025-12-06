import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import TD3Actor, TD3Critic
from buffer import ReplayBuffer

class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient Agent"""
    def __init__(self, obs_dim, action_dim, max_action, hyperparameters, device):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Hyperparameters
        self.gamma = hyperparameters.get('gamma', 0.99)
        self.tau = hyperparameters.get('tau', 0.005)
        self.lr = hyperparameters.get('lr', 3e-4)
        self.policy_noise = hyperparameters.get('policy_noise', 0.2)
        self.noise_clip = hyperparameters.get('noise_clip', 0.5)
        self.policy_freq = hyperparameters.get('policy_freq', 2)
        self.expl_noise = hyperparameters.get('expl_noise', 0.1)
        self.hidden_dim = hyperparameters.get('hidden_dim', 256)
        
        # Actor network and target
        self.actor = TD3Actor(obs_dim, action_dim, self.hidden_dim, max_action).to(device)
        self.actor_target = TD3Actor(obs_dim, action_dim, self.hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        
        # Twin Critic networks and targets
        self.critic = TD3Critic(obs_dim, action_dim, self.hidden_dim).to(device)
        self.critic_target = TD3Critic(obs_dim, action_dim, self.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.total_it = 0
    
    def select_action(self, state, eval_mode=False):
        """Select action from policy with exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if not eval_mode:
            # Add exploration noise
            noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def update(self, batch):
        """Update TD3 networks"""
        self.total_it += 1
        
        states, actions, rewards, next_states, dones = batch
        
        # ==================== Update Critic ====================
        with torch.no_grad():
            # Select action from target policy and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Compute target Q-values using twin critics
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ==================== Delayed Policy Update ====================
        policy_loss = None
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            policy_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            
            # ==================== Soft Update Target Networks ====================
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item() if policy_loss is not None else 0.0
        }
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
