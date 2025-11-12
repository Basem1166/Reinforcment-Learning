import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
from model import QNetwork, ReplayMemory, Transition

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    
    def __init__(self, state_dim, action_dim, memory_size, batch_size,
                 lr, gamma, epsilon_decay, epsilon_min, is_ddqn=False,
                 layer1_size=128, layer2_size=128):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0  # Start epsilon at 1.0
        self.epsilon_decay_factor = epsilon_decay # This is now our multiplicative factor
        self.epsilon_min = epsilon_min
        self.is_ddqn = is_ddqn

        # Create two networks using the new size arguments
        self.policy_net = QNetwork(state_dim, action_dim, layer1_size, layer2_size).to(device)
        self.target_net = QNetwork(state_dim, action_dim, layer1_size, layer2_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is in evaluation mode

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        Epsilon is no longer calculated here.
        """
        if random.random() > self.epsilon:
            # Exploit: Choose the best action from the policy network
            with torch.no_grad():
                # state is already a tensor from main.py, just add batch dim
                state = state.unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].view(1, 1) # .max(1)[1] gets the index (action)
        else:
            # Explore: Choose a random action
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)

    def update_epsilon(self):
        """
        Multiplicatively decays epsilon, called at the end of each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_factor)

    def optimize_model(self):
        """
        Performs one step of optimization on the policy network.
        """
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples in memory

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch
        batch = Transition(*zip(*transitions))

        # Create tensors from the batch
        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        next_state_batch = torch.stack(batch.next_state).to(device)
        done_batch = torch.tensor(batch.done, device=device, dtype=torch.float32)

        # 1. Get Q(s, a) for the current states
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # 2. Calculate the target Q-value (V(s'))
        with torch.no_grad():
            if self.is_ddqn:
                # --- DDQN Target ---
                policy_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(-1)
                next_q_values = self.target_net(next_state_batch).gather(1, policy_actions)
                next_q_values = next_q_values.squeeze(-1)
            else:
                # --- Standard DQN Target ---
                next_q_values = self.target_net(next_state_batch).max(1)[0]
            
            # 3. Calculate the expected Q-value (the "target")
            target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        # 4. Compute the loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))

        # 5. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()