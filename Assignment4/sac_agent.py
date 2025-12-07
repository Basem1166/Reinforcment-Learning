import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import SACQNetwork, SACGaussianPolicy
from buffer import ReplayBuffer

class SACAgent:
    """Soft Actor-Critic Agent for continuous action spaces"""
    def __init__(self, obs_dim, action_dim, action_scale, hyperparameters, device):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        # Hyperparameters
        self.gamma = hyperparameters.get('gamma', 0.99)
        self.tau = hyperparameters.get('tau', 0.005)
        self.lr = hyperparameters.get('lr', 3e-4)
        self.alpha = hyperparameters.get('alpha', 0.2)
        self.auto_entropy = hyperparameters.get('auto_entropy', True)
        self.hidden_dim = hyperparameters.get('hidden_dim', 256)
        
        # Policy network
        self.policy = SACGaussianPolicy(
            obs_dim, action_dim, self.hidden_dim, action_scale
        ).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Twin Q-networks
        self.q1 = SACQNetwork(obs_dim, action_dim, self.hidden_dim).to(device)
        self.q2 = SACQNetwork(obs_dim, action_dim, self.hidden_dim).to(device)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.lr)
        
        # Target Q-networks
        self.q1_target = SACQNetwork(obs_dim, action_dim, self.hidden_dim).to(device)
        self.q2_target = SACQNetwork(obs_dim, action_dim, self.hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Automatic entropy tuning
        if self.auto_entropy:
            self.target_entropy = -action_dim  # Heuristic: -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
            self.alpha = self.log_alpha.exp().item()
    
    def select_action(self, state, eval_mode=False):
        """Select action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if eval_mode:
                # Use mean action for evaluation
                mean, _ = self.policy(state)
                action = mean
            else:
                # Sample action during training
                action, _, _ = self.policy.sample(state)
        
        return action.cpu().numpy()[0]
    
    def update(self, batch):
        """Update SAC networks"""
        states, actions, rewards, next_states, dones = batch
        
        # ==================== Update Q-Networks ====================
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            
            # Compute target Q-values using twin Q-networks
            q1_target_next = self.q1_target(next_states, next_actions)
            q2_target_next = self.q2_target(next_states, next_actions)
            min_q_target_next = torch.min(q1_target_next, q2_target_next)
            
            # Add entropy term
            next_q_value = min_q_target_next - self.alpha * next_log_probs
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value
        
        # Current Q-values
        q1_value = self.q1(states, actions)
        q2_value = self.q2(states, actions)
        
        # Q-loss
        q1_loss = F.mse_loss(q1_value, target_q_value)
        q2_loss = F.mse_loss(q2_value, target_q_value)
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # ==================== Update Policy ====================
        # Sample new actions from current policy
        new_actions, log_probs, _ = self.policy.sample(states)
        
        # Q-values for new actions
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Policy loss
        policy_loss = (self.alpha * log_probs - min_q_new).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ==================== Update Alpha (Temperature) ====================
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # ==================== Soft Update Target Networks ====================
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha
        }
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
