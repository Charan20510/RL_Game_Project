"""Replay buffers and rollout storage for RL algorithms.

Includes:
- RolloutBuffer: on-policy storage for PPO with GAE computation
- ReplayBuffer: simple circular buffer for DQN
- PrioritizedReplayBuffer: Sum-Tree based PER for Rainbow DQN
- NStepReplayBuffer: wraps PER with n-step return accumulation
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Rollout Buffer (PPO — on-policy)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-size buffer for on-policy rollout collection.

    Stores transitions and computes GAE advantages + discounted returns
    after the rollout is complete. Also stores action masks for PPO updates.
    """

    def __init__(self, rollout_length: int, obs_dim: int, n_actions: int, gamma: float, gae_lambda: float) -> None:
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = np.zeros((rollout_length, obs_dim), dtype=np.int16)
        self.actions = np.zeros(rollout_length, dtype=np.int64)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.float32)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.advantages = np.zeros(rollout_length, dtype=np.float32)
        self.returns = np.zeros(rollout_length, dtype=np.float32)
        self.action_masks = np.ones((rollout_length, n_actions), dtype=np.bool_)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        action_mask: np.ndarray | None = None,
    ) -> None:
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        if action_mask is not None:
            self.action_masks[self.ptr] = action_mask
        self.ptr += 1
        if self.ptr >= self.rollout_length:
            self.full = True

    def compute_gae(self, last_value: float, last_done: bool) -> None:
        """Compute Generalized Advantage Estimation in-place."""
        gae = 0.0
        size = self.ptr  # might not be full
        for t in reversed(range(size)):
            if t == size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def get_batches(self, minibatch_size: int):
        """Yield random minibatch indices for PPO epoch updates."""
        size = self.ptr
        indices = np.random.permutation(size)
        for start in range(0, size, minibatch_size):
            end = min(start + minibatch_size, size)
            batch_idx = indices[start:end]
            yield {
                "observations": self.observations[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "advantages": self.advantages[batch_idx],
                "returns": self.returns[batch_idx],
                "values": self.values[batch_idx],
                "action_masks": self.action_masks[batch_idx],
            }

    def reset(self) -> None:
        self.ptr = 0
        self.full = False


# ---------------------------------------------------------------------------
# Sum Tree for Prioritized Replay
# ---------------------------------------------------------------------------

class _SumTree:
    """Binary sum-tree for efficient O(log n) proportional sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float) -> int:
        """Add a new entry (or overwrite oldest). Returns data index."""
        tree_idx = self.data_ptr + self.capacity - 1
        self.update(tree_idx, priority)
        data_idx = self.data_ptr
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def get(self, cumsum: float) -> Tuple[int, int, float]:
        """Find the leaf node for a given cumulative sum.

        Returns (tree_idx, data_idx, priority).
        """
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if cumsum <= self.tree[left]:
                idx = left
            else:
                cumsum -= self.tree[left]
                idx = right

        data_idx = idx - self.capacity + 1
        return idx, data_idx, self.tree[idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        leaf_start = self.capacity - 1
        return float(np.max(self.tree[leaf_start:leaf_start + self.size])) if self.size > 0 else 1.0


# ---------------------------------------------------------------------------
# Prioritized Experience Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Experience replay with priority-based sampling using a Sum-Tree.

    Supports importance-sampling weight computation with beta annealing.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 100_000,
        epsilon: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        self._step = 0

        self.tree = _SumTree(capacity)

        # Pre-allocate numpy arrays for zero-copy storage
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    @property
    def beta(self) -> float:
        frac = min(1.0, self._step / max(1, self.beta_anneal_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        priority = self.tree.max_priority ** self.alpha
        if priority == 0.0:
            priority = 1.0
        data_idx = self.tree.add(priority)
        self.obs[data_idx] = obs
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_obs[data_idx] = next_obs
        self.dones[data_idx] = float(done)

    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Sample a prioritised minibatch.

        Returns:
            batch: dict with obs, actions, rewards, next_obs, dones
            indices: tree indices for priority updates
            weights: importance-sampling weights (normalised)
        """
        self._step += 1
        beta = self.beta

        total = self.tree.total
        segment = total / batch_size

        indices = np.zeros(batch_size, dtype=np.int64)
        data_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            cumsum = np.random.uniform(low, high)
            tree_idx, d_idx, prio = self.tree.get(cumsum)
            indices[i] = tree_idx
            data_indices[i] = d_idx
            priorities[i] = max(prio, self.epsilon)

        # Importance-sampling weights
        probs = priorities / max(total, self.epsilon)
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()  # Normalise

        batch = {
            "obs": self.obs[data_indices],
            "actions": self.actions[data_indices],
            "rewards": self.rewards[data_indices],
            "next_obs": self.next_obs[data_indices],
            "dones": self.dones[data_indices],
        }

        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, prio in zip(indices, priorities):
            self.tree.update(int(idx), float(prio))

    def __len__(self) -> int:
        return self.tree.size


# ---------------------------------------------------------------------------
# N-Step Replay Buffer (wraps PER)
# ---------------------------------------------------------------------------

class NStepReplayBuffer:
    """Accumulates n-step returns before inserting into a PrioritizedReplayBuffer.

    Computes: R_n = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1}
    and stores (s_t, a_t, R_n, s_{t+n}, done_{t+n}).
    """

    def __init__(
        self,
        per_buffer: PrioritizedReplayBuffer,
        n_step: int = 3,
        gamma: float = 0.99,
    ) -> None:
        self.per_buffer = per_buffer
        self.n_step = n_step
        self.gamma = gamma

        # Temporary transition deque
        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[int] = []
        self._rew_buf: list[float] = []
        self._next_obs_buf: list[np.ndarray] = []
        self._done_buf: list[bool] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._obs_buf.append(obs)
        self._act_buf.append(action)
        self._rew_buf.append(reward)
        self._next_obs_buf.append(next_obs)
        self._done_buf.append(done)

        # If episode ended or buffer full, flush
        if done:
            self._flush_all()
        elif len(self._obs_buf) >= self.n_step:
            self._flush_one()

    def _flush_one(self) -> None:
        """Flush the oldest transition as an n-step transition."""
        n = min(len(self._obs_buf), self.n_step)
        # Compute n-step return
        n_step_return = 0.0
        for i in range(n):
            n_step_return += (self.gamma ** i) * self._rew_buf[i]
            if self._done_buf[i] and i < n - 1:
                # Episode ended before n steps; truncate
                n = i + 1
                break

        obs = self._obs_buf[0]
        action = self._act_buf[0]
        next_obs = self._next_obs_buf[n - 1]
        done = self._done_buf[n - 1]

        self.per_buffer.add(obs, action, n_step_return, next_obs, done)

        # Remove oldest
        self._obs_buf.pop(0)
        self._act_buf.pop(0)
        self._rew_buf.pop(0)
        self._next_obs_buf.pop(0)
        self._done_buf.pop(0)

    def _flush_all(self) -> None:
        """Flush all remaining transitions (end of episode)."""
        while len(self._obs_buf) > 0:
            self._flush_one()

    def sample(self, batch_size: int):
        return self.per_buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.per_buffer.update_priorities(indices, td_errors)

    def __len__(self) -> int:
        return len(self.per_buffer)
