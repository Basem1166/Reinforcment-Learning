import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model_universal import UniversalActor, UniversalCritic

class PPOAgent:
    def __init__(self, env, hyperparameters):
        self.env = env
        self.gamma = hyperparameters["gamma"]
        self.lr = hyperparameters["learning_rate"]
        self.clip = hyperparameters.get("ppo_clip", 0.2)
        self.k_epochs = 10
        self.entropy_coef = hyperparameters.get("entropy_coef", 0.01)
        
        self.obs_dim = env.observation_space.shape[0]
        
        # Separate Networks
        self.actor = UniversalActor(self.obs_dim, env.action_space)
        self.critic = UniversalCritic(self.obs_dim)
        
        # Separate Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.memory = []

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            dist = self.actor.get_distribution(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        if self.actor.is_continuous:
            log_prob = log_prob.sum(dim=-1)
            action_val = action.numpy()[0]
            action_val = np.clip(action_val, self.env.action_space.low, self.env.action_space.high)
            return action_val, log_prob.item()
        else:
            return action.item(), log_prob.item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def update(self):
        states, actions, log_probs_old, rewards, dones, next_states = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states))
        
        if self.actor.is_continuous:
            actions = torch.FloatTensor(np.array(actions))
        else:
            actions = torch.LongTensor(np.array(actions))

        log_probs_old = torch.FloatTensor(np.array(log_probs_old))
        rewards = torch.FloatTensor(np.array(rewards))
        dones = torch.FloatTensor(np.array(dones))
        
        # 1. Rewards-to-Go
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns)
        
        # 2. Advantages
        # We calculate this once outside the epoch loop
        values = self.critic(states).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Update Loop
        for _ in range(self.k_epochs):
            # --- Actor Update ---
            dist = self.actor.get_distribution(states)
            log_probs_new = dist.log_prob(actions)
            
            if self.actor.is_continuous:
                log_probs_new = log_probs_new.sum(dim=-1)
                
            entropy = dist.entropy().mean()
            ratio = torch.exp(log_probs_new - log_probs_old)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # --- Critic Update ---
            value_preds = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(value_preds, returns)
            
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        self.memory = []
        return actor_loss.item() + critic_loss.item()