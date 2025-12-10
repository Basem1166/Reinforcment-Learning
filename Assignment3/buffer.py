import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Samples a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SumTree:
    """
    Sum Tree data structure for efficient priority-based sampling.
    Stores priorities in leaves and maintains partial sums in parent nodes.
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (transitions)
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree array
        self.data = np.zeros(capacity, dtype=object)  # Transition storage
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find leaf index for a given cumulative sum s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return total priority sum."""
        return self.tree[0]

    def add(self, priority, data):
        """Add transition with given priority."""
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx, priority):
        """Update priority at tree index."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        """Get transition for cumulative sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Prioritizes transitions with:
    1. High TD errors (learns from surprising experiences)
    2. High rewards (learns from successes - especially important for MountainCar)
    
    Reference: Schaul et al., "Prioritized Experience Replay" (2016)
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001,
                 reward_priority_weight=0.5, success_bonus=10.0):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (annealed to 1)
            beta_increment: How much to increase beta each sample call
            reward_priority_weight: Weight for reward-based priority (0-1)
            success_bonus: Extra priority bonus for successful episodes (MountainCar: reaching goal)
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to ensure non-zero priority
        self.max_priority = 1.0
        
        # For MountainCar success prioritization
        self.reward_priority_weight = reward_priority_weight
        self.success_bonus = success_bonus

    def push(self, state, action, reward, next_state, done, td_error=None):
        """
        Add transition with priority based on TD error and reward.
        For MountainCar, reaching the goal (reward > -1) gets bonus priority.
        """
        transition = (state, action, reward, next_state, done)
        
        # Compute priority
        if td_error is not None:
            td_priority = (np.abs(td_error) + self.epsilon) ** self.alpha
        else:
            td_priority = self.max_priority
        
        # Reward-based priority (especially for sparse reward envs like MountainCar)
        # MountainCar: reward is -1 per step, success gives higher reward or done=True at goal
        reward_priority = 0.0
        if reward > -1.0:  # MountainCar success (reached goal)
            reward_priority = self.success_bonus
        elif reward > -0.5:  # For other envs with positive rewards
            reward_priority = (reward + 1.0) ** self.alpha
        
        # Combined priority
        priority = (1 - self.reward_priority_weight) * td_priority + \
                   self.reward_priority_weight * (reward_priority + self.epsilon) ** self.alpha
        
        # Ensure minimum priority
        priority = max(priority, self.epsilon)
        
        self.max_priority = max(self.max_priority, priority)
        self.tree.add(priority, transition)

    def sample(self, batch_size):
        """
        Sample batch with probability proportional to priority.
        Returns transitions, indices (for updating), and importance sampling weights.
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            if data is None or (isinstance(data, int) and data == 0):
                # Handle edge case: sample random valid transition
                s = random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        priorities = np.array(priorities)
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Unpack batch
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return state, action, reward, next_state, done, indices, weights

    def update_priorities(self, indices, td_errors, rewards=None):
        """
        Update priorities after learning.
        Can incorporate both TD errors and rewards.
        """
        for i, (idx, td_error) in enumerate(zip(indices, td_errors)):
            td_priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            
            # Add reward bonus if provided
            if rewards is not None:
                reward = rewards[i]
                if reward > -1.0:  # Success
                    reward_priority = self.success_bonus
                else:
                    reward_priority = 0.0
                priority = (1 - self.reward_priority_weight) * td_priority + \
                           self.reward_priority_weight * (reward_priority + self.epsilon) ** self.alpha
            else:
                priority = td_priority
            
            priority = max(priority, self.epsilon)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries