import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import PPOActor, PPOCritic
from buffer import RolloutBuffer

class PPOAgent:
    """Proximal Policy Optimization Agent for continuous action spaces"""
    def __init__(self, obs_dim, action_dim, hyperparameters, device):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = hyperparameters.get('gamma', 0.99)
        self.gae_lambda = hyperparameters.get('gae_lambda', 0.95)
        self.lr = hyperparameters.get('lr', 3e-4)
        self.clip_epsilon = hyperparameters.get('clip_epsilon', 0.2)
        self.value_loss_coef = hyperparameters.get('value_loss_coef', 0.5)
        self.entropy_coef = hyperparameters.get('entropy_coef', 0.01)
        self.max_grad_norm = hyperparameters.get('max_grad_norm', 0.5)
        self.ppo_epochs = hyperparameters.get('ppo_epochs', 10)
        self.mini_batch_size = hyperparameters.get('mini_batch_size', 64)
        self.hidden_dim = hyperparameters.get('hidden_dim', 256)
        
        # Networks
        self.actor = PPOActor(obs_dim, action_dim, self.hidden_dim).to(device)
        self.critic = PPOCritic(obs_dim, self.hidden_dim).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=self.lr
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
    
    def select_action(self, state, eval_mode=False):
        """Select action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if eval_mode:
                # Use mean action for evaluation
                mean, _ = self.actor(state)
                action = mean
                log_prob = None
                value = None
            else:
                # Sample action during training
                dist = self.actor.get_dist(state)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.critic(state)
        
        if eval_mode:
            return action.cpu().numpy()[0]
        else:
            return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """Store transition in rollout buffer"""
        self.buffer.add(state, action, reward, done, log_prob, value)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, next_state):
        """Update PPO networks"""
        states, actions, rewards, dones, old_log_probs, values = self.buffer.get()
        
        # Get next value for GAE computation
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.critic(next_state_tensor).cpu().item()
        
        # Compute advantages using GAE
        advantages = self.compute_gae(rewards, values, dones, next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        # PPO update for multiple epochs
        dataset_size = states.size(0)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        
        for _ in range(self.ppo_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > dataset_size:
                    continue
                
                idx = indices[start:end]
                
                # Mini-batch data
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                # Evaluate actions
                dist = self.actor.get_dist(mb_states)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Value prediction
                values_pred = self.critic(mb_states).squeeze()
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values_pred, mb_returns)
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Update networks
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / update_count,
            'value_loss': total_value_loss / update_count,
            'entropy': total_entropy / update_count
        }
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
