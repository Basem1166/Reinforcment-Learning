import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model_sac import QNetwork, GaussianPolicy

class SACAgent:
    def __init__(self, env, hyperparameters):
        self.gamma = hyperparameters["gamma"]
        self.tau = hyperparameters["tau"] # Soft update parameter (rho in some slides)
        self.alpha = hyperparameters["entropy_coef"] # Fixed entropy coefficient
        self.lr = hyperparameters["learning_rate"]
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Scale actions (e.g. Pendulum has range [-2, 2])
        self.action_scale = torch.FloatTensor(
            (env.action_space.high - env.action_space.low) / 2.)
        
        # ---------------------------------------------------
        # 1. Initialize Networks (Step 1)
        # ---------------------------------------------------
        
        # Critics (Double Q-Learning): phi1, phi2
        self.q1 = QNetwork(self.obs_dim, self.action_dim)
        self.q2 = QNetwork(self.obs_dim, self.action_dim)
        
        # Actor (Policy): theta
        self.policy = GaussianPolicy(self.obs_dim, self.action_dim, self.action_scale)
        
        # Target Networks (Step 2)
        self.q1_target = QNetwork(self.obs_dim, self.action_dim)
        self.q2_target = QNetwork(self.obs_dim, self.action_dim)
        
        # Hard copy weights initially
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=self.lr)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr)

    def get_action(self, state, evaluate=False):
        """
        Selects an action. 
        If evaluate=True, returns the Mean (deterministic).
        If evaluate=False, samples from the distribution (stochastic).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if evaluate:
            _, _, mean = self.policy.sample(state_tensor)
            return mean.detach().numpy()[0]
        else:
            action, _, _ = self.policy.sample(state_tensor)
            return action.detach().numpy()[0]

    def update(self, memory, batch_size):
        # Step 11: Randomly sample a batch of transitions
        state, action, reward, next_state, done = memory.sample(batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        # ---------------------------------------------------
        # Step 12: Compute Targets for Q Functions
        # ---------------------------------------------------
        with torch.no_grad():
            # Sample next action from CURRENT policy
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state)
            
            # Get target Q-values
            qf1_next_target = self.q1_target(next_state, next_state_action)
            qf2_next_target = self.q2_target(next_state, next_state_action)
            
            # Select Minimum Q (Double Q-Learning Trick)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) 
            
            # Apply Entropy Term: V = Q - alpha * log_pi
            min_qf_next_target = min_qf_next_target - self.alpha * next_state_log_pi
            
            # Compute Bellman Target: y = r + gamma * (1-d) * V
            next_q_value = reward + (1 - done) * self.gamma * min_qf_next_target

        # ---------------------------------------------------
        # Step 13: Update Q-Functions (Critics)
        # ---------------------------------------------------
        # Predictions
        qf1 = self.q1(state, action)
        qf2 = self.q2(state, action)
        
        # MSE Loss
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        
        # Optimize Q1
        self.q1_optim.zero_grad()
        qf1_loss.backward()
        self.q1_optim.step()
        
        # Optimize Q2
        self.q2_optim.zero_grad()
        qf2_loss.backward()
        self.q2_optim.step()

        # ---------------------------------------------------
        # Step 14: Update Policy (Actor)
        # ---------------------------------------------------
        # Sample action from current policy using Reparameterization
        pi, log_pi, _ = self.policy.sample(state)
        
        # Get Q-values for the sampled actions (using the MAIN Q networks)
        qf1_pi = self.q1(state, pi)
        qf2_pi = self.q2(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        # Policy Loss: alpha * log_pi - min_Q
        # (We want to Maximize Q and Maximize Entropy, so we Minimize the negative)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # ---------------------------------------------------
        # Step 15: Soft Update Target Networks
        # ---------------------------------------------------
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return policy_loss.item() + qf1_loss.item() + qf2_loss.item()