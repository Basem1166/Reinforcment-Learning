import torch
import torch.optim as optim
import numpy as np
from model_universal import UniversalActor, UniversalCritic

class A2CAgent:
    def __init__(self, env, hyperparameters):
        self.env = env
        self.gamma = hyperparameters["gamma"]
        self.lr = hyperparameters["learning_rate"]
        self.entropy_coef = hyperparameters.get("entropy_coef", 0.01)
        
        self.obs_dim = env.observation_space.shape[0]
        
        # --- SEPARATE NETWORKS ---
        self.actor = UniversalActor(self.obs_dim, env.action_space)
        self.critic = UniversalCritic(self.obs_dim)
        
        # --- SEPARATE OPTIMIZERS ---
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        dist = self.actor.get_distribution(state_tensor)
        action = dist.sample()
        
        if self.actor.is_continuous:
            action_val = action.detach().numpy()[0]
            action_val = np.clip(action_val, self.env.action_space.low, self.env.action_space.high)
            return action_val
        else:
            return action.item()

    def update(self, rollouts):
        if len(rollouts) == 0:
            return

        states, actions, rewards, next_states, dones = zip(*rollouts)
        
        states = torch.FloatTensor(np.array(states))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        if self.actor.is_continuous:
            actions = torch.FloatTensor(np.array(actions))
        else:
            actions = torch.LongTensor(np.array(actions))

        # ----------------------------
        # 1. Critic Update
        # ----------------------------
        V_val = self.critic(states)
        V_next = self.critic(next_states)
        
        # Target = r + gamma * V(next) * (1-done)
        V_target = rewards + self.gamma * V_next.detach() * (1 - dones)
        
        # Advantage = Target - V(s)
        # Note: We calculate advantage here for the critic loss
        advantage = V_target - V_val
        critic_loss = 0.5 * advantage.pow(2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ----------------------------
        # 2. Actor Update
        # ----------------------------
        # Re-calculate advantage (detached from critic) for Actor
        # Or reuse the one above but detached
        actor_advantage = (V_target - V_val).detach()
        
        dist = self.actor.get_distribution(states)
        log_prob = dist.log_prob(actions)
        
        if self.actor.is_continuous:
            log_prob = log_prob.sum(dim=-1).unsqueeze(1)
        elif log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(1)
            
        # Loss = -log_prob * advantage
        actor_loss = -(log_prob * actor_advantage).mean()
        entropy_loss = dist.entropy().mean()

        total_actor_loss = actor_loss - self.entropy_coef * entropy_loss
        
        self.actor_optim.zero_grad()
        total_actor_loss.backward()
        self.actor_optim.step()
        
        return critic_loss.item() + total_actor_loss.item()