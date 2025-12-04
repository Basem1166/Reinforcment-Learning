import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from model_sac import QNetwork, GaussianPolicy, DiscreteQNetwork, CategoricalPolicy

class SACAgent:
    def __init__(self, env, hyperparameters):
        self.gamma = hyperparameters["gamma"]
        self.tau = hyperparameters["tau"] # Soft update parameter (rho in some slides)
        self.lr = hyperparameters["learning_rate"]
        
        # Automatic entropy tuning (if enabled)
        self.auto_entropy = hyperparameters.get("auto_entropy", True)
        
        self.obs_dim = env.observation_space.shape[0]
        
        # Auto-detect discrete vs continuous action space
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        
        if self.is_discrete:
            # Discrete action space (CartPole, Acrobot, MountainCar)
            self.action_dim = env.action_space.n
            print(f"SAC-Discrete: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
            
            # Q-networks output Q-values for all actions
            self.q1 = DiscreteQNetwork(self.obs_dim, self.action_dim)
            self.q2 = DiscreteQNetwork(self.obs_dim, self.action_dim)
            self.q1_target = DiscreteQNetwork(self.obs_dim, self.action_dim)
            self.q2_target = DiscreteQNetwork(self.obs_dim, self.action_dim)
            
            # Categorical policy
            self.policy = CategoricalPolicy(self.obs_dim, self.action_dim)
            
            # Target entropy for discrete: -log(1/|A|) * ratio = log(|A|) * ratio
            # Common choice: 0.98 * log(|A|) to encourage near-uniform exploration
            self.target_entropy = 0.98 * np.log(self.action_dim)
        else:
            # Continuous action space (Pendulum, MountainCarContinuous)
            self.action_dim = env.action_space.shape[0]
            print(f"SAC-Continuous: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
            
            # Scale actions (e.g. Pendulum has range [-2, 2])
            self.action_scale = torch.FloatTensor(
                (env.action_space.high - env.action_space.low) / 2.)
            
            # Q-networks take state+action as input
            self.q1 = QNetwork(self.obs_dim, self.action_dim)
            self.q2 = QNetwork(self.obs_dim, self.action_dim)
            self.q1_target = QNetwork(self.obs_dim, self.action_dim)
            self.q2_target = QNetwork(self.obs_dim, self.action_dim)
            
            # Gaussian policy
            self.policy = GaussianPolicy(self.obs_dim, self.action_dim, self.action_scale)
            
            # Target entropy for continuous: -dim(A) (standard heuristic)
            self.target_entropy = -self.action_dim
        
        # Initialize alpha (entropy coefficient)
        if self.auto_entropy:
            # Learnable log_alpha for numerical stability
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
            print(f"Auto-entropy tuning enabled. Target entropy: {self.target_entropy:.4f}")
        else:
            self.alpha = hyperparameters["entropy_coef"]
            self.log_alpha = None
            print(f"Fixed entropy coefficient: {self.alpha}")
        
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
        If evaluate=True, returns deterministic action.
        If evaluate=False, samples from the distribution (stochastic).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if self.is_discrete:
            if evaluate:
                # Deterministic: pick action with highest probability
                probs = self.policy.get_probs(state_tensor)
                return probs.argmax(dim=-1).item()
            else:
                # Stochastic: sample from categorical
                action, _, _ = self.policy.sample(state_tensor)
                return action.item()
        else:
            # Continuous
            if evaluate:
                _, _, mean = self.policy.sample(state_tensor)
                return mean.detach().numpy()[0]
            else:
                action, _, _ = self.policy.sample(state_tensor)
                return action.detach().numpy()[0]

    def update(self, memory, batch_size):
        """
        Update networks. Supports both regular and prioritized replay buffers.
        """
        # Check if using prioritized replay
        is_per = hasattr(memory, 'update_priorities')
        
        if is_per:
            state, action, reward, next_state, done, indices, weights = memory.sample(batch_size)
            weights = torch.FloatTensor(weights).unsqueeze(1)
        else:
            state, action, reward, next_state, done = memory.sample(batch_size)
            indices, weights = None, None
        
        state = torch.FloatTensor(state)
        reward_tensor = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        if self.is_discrete:
            td_errors = self._update_discrete(state, action, reward_tensor, next_state, done, weights)
        else:
            action = torch.FloatTensor(action)
            td_errors = self._update_continuous(state, action, reward_tensor, next_state, done, weights)
        
        # Update priorities if using PER
        if is_per and td_errors is not None:
            memory.update_priorities(indices, td_errors.detach().cpu().numpy(), reward)
        
        return td_errors.abs().mean().item() if td_errors is not None else 0.0
    
    def _update_discrete(self, state, action, reward, next_state, done, weights=None):
        """SAC-Discrete update (Christodoulou, 2019)"""
        action = torch.LongTensor(action)
        
        # ---------------------------------------------------
        # Compute Targets for Q Functions
        # ---------------------------------------------------
        with torch.no_grad():
            # Get action probabilities and log probs for next state
            next_probs, next_log_probs = self.policy.evaluate(next_state)
            
            # Q-values from target networks for all actions
            q1_next = self.q1_target(next_state)
            q2_next = self.q2_target(next_state)
            min_q_next = torch.min(q1_next, q2_next)
            
            # V(s') = sum_a [ pi(a|s') * (Q(s',a) - alpha * log pi(a|s')) ]
            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            
            # Bellman target
            target_q = reward + (1 - done) * self.gamma * v_next
        
        # ---------------------------------------------------
        # Update Q-Functions
        # ---------------------------------------------------
        # Get Q-values for taken actions
        q1_all = self.q1(state)
        q2_all = self.q2(state)
        q1 = q1_all.gather(1, action.unsqueeze(1))
        q2 = q2_all.gather(1, action.unsqueeze(1))
        
        # TD errors for prioritized replay
        td_errors = (q1 - target_q).detach().squeeze()
        
        # Apply importance sampling weights if using PER
        if weights is not None:
            q1_loss = (weights * (q1 - target_q).pow(2)).mean()
            q2_loss = (weights * (q2 - target_q).pow(2)).mean()
        else:
            q1_loss = F.mse_loss(q1, target_q)
            q2_loss = F.mse_loss(q2, target_q)
        
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()
        
        # ---------------------------------------------------
        # Update Policy
        # ---------------------------------------------------
        probs, log_probs = self.policy.evaluate(state)
        
        q1_pi = self.q1(state)
        q2_pi = self.q2(state)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        # Policy loss: sum_a [ pi(a|s) * (alpha * log pi(a|s) - Q(s,a)) ]
        policy_loss = (probs * (self.alpha * log_probs - min_q_pi)).sum(dim=-1).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        # ---------------------------------------------------
        # Update Alpha (Entropy Coefficient)
        # ---------------------------------------------------
        if self.auto_entropy:
            # Entropy of current policy: H = -sum_a [ pi(a|s) * log pi(a|s) ]
            with torch.no_grad():
                entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # Alpha loss: alpha * (entropy - target_entropy)
            alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # ---------------------------------------------------
        # Soft Update Target Networks
        # ---------------------------------------------------
        self._soft_update()
        
        return td_errors
    
    def _update_continuous(self, state, action, reward, next_state, done, weights=None):
        """Original continuous SAC update"""
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
        # Update Q-Functions (Critics)
        # ---------------------------------------------------
        # Predictions
        qf1 = self.q1(state, action)
        qf2 = self.q2(state, action)
        
        # TD errors for prioritized replay
        td_errors = (qf1 - next_q_value).detach().squeeze()
        
        # Apply importance sampling weights if using PER
        if weights is not None:
            qf1_loss = (weights * (qf1 - next_q_value).pow(2)).mean()
            qf2_loss = (weights * (qf2 - next_q_value).pow(2)).mean()
        else:
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
        # Update Policy (Actor)
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
        # Update Alpha (Entropy Coefficient)
        # ---------------------------------------------------
        if self.auto_entropy:
            # Alpha loss: E[-log_pi - target_entropy] * alpha
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.exp().item()

        # Soft Update Target Networks
        self._soft_update()

        return td_errors
    
    def _soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)