import numpy as np
import torch


class ReplayBuffer:
    """Replay buffer that supports both vector and image observations.

    For vector states, pass obs_shape as an int (dimension).
    For image states (e.g., CarRacing 96x96x3), pass obs_shape as a tuple.
    """

    def __init__(self, obs_shape, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        if isinstance(obs_shape, int):
            state_shape = (max_size, obs_shape)
        else:
            state_shape = (max_size, *obs_shape)

        self.states = np.zeros(state_shape, dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros(state_shape, dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.as_tensor(self.states[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_states[ind], dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[ind], dtype=torch.float32, device=device),
        )

    def __len__(self):
        return self.size


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms (PPO)"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.dones,
            self.log_probs,
            self.values
        )
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.states)
