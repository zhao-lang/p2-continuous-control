from collections import namedtuple
import random
import copy

import segment_tree

import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = []
        self.memory_idx = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.memory_idx] = e
        self.memory_idx += 1
        self.memory_idx = self.memory_idx % self.buffer_size
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        return self._tensors_from_experiences(experiences)
    
    def _tensors_from_experiences(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples with priority."""

    def __init__(self, action_size, buffer_size, batch_size, seed, a):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            a (float): sampling parameter
        """
        assert a >= 0 and a <= 1.0
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)
        
        self.max_priority = 1.0
        self.tree_idx = 0
        self.a = a

        capacity = 1
        while capacity < buffer_size:
            capacity *= 2

        self.sum_tree = segment_tree.SumSegmentTree(capacity)
        self.min_tree = segment_tree.MinSegmentTree(capacity)
    
    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)

        self.sum_tree[self.tree_idx] = self.max_priority ** self.a
        self.min_tree[self.tree_idx] = self.max_priority ** self.a
        self.tree_idx += 1
        self.tree_idx = self.tree_idx % self.buffer_size

    def sample(self, beta=0.5):
        assert len(self.memory) >= self.batch_size
        assert beta >= 0 and beta <= 1.0

        idxs = self._sample_with_priority()
        experiences = [self.memory[idx] for idx in idxs]
        tensors = self._tensors_from_experiences(experiences)
        weights = torch.from_numpy(np.array([self._calc_weight(idx, beta) for idx in idxs])).float().to(device)

        return tensors + (weights, idxs,)
    
    def _sample_with_priority(self):
        idxs = []

        total_priority = self.sum_tree.sum(0, len(self.memory))
        segment_priority = total_priority / self.batch_size

        for i in range(self.batch_size):
            low = segment_priority * i
            high = low + segment_priority
            prefix_sum = random.uniform(low, high)
            idx = self.sum_tree.find_prefixsum_idx(prefix_sum)
            idxs.append(idx)

        return idxs
    
    def _calc_weight(self, idx, beta):
        min_priority = self.min_tree.min(0, len(self.memory)) / self.sum_tree.sum(0, len(self.memory))
        max_weight = (min_priority * len(self.memory)) ** (-1 * beta)
        
        priority = self.sum_tree[idx] / self.sum_tree.sum(0, len(self.memory))
        weight = (priority * len(self.memory)) ** (-1 * beta)
        weight = weight / max_weight
        
        return weight
    
    def update_priority(self, idx, priority):
        assert priority > 0
        assert idx >= 0 and idx < len(self.memory)

        self.sum_tree[idx] = priority ** self.a
        self.min_tree[idx] = priority ** self.a

        self.max_priority = max(self.max_priority, priority)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)

        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state