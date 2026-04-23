"""Test-time Monte Carlo Tree Search (MCTS) evaluator for Bobby Carrot.

Uses the real game environment as a simulator and the trained PPO value head
as a leaf evaluator. No learned dynamics model required — transitions are fully
deterministic, so the true env is the perfect world model.

Usage:
    evaluator = MCTSEvaluator(agent, preprocessor, n_sims=128, depth=25)
    action = evaluator.select_action(env, obs_raw)
"""

from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .ppo import PPOAgent
from .networks import ObservationPreprocessor


class _MCTSNode:
    """A single node in the MCTS tree."""

    __slots__ = (
        "action",        # action that led to this node from parent
        "parent",
        "children",
        "visit_count",
        "total_value",
        "prior",         # policy prior P(a|s) from PPO policy head
        "is_terminal",
    )

    def __init__(
        self,
        action: Optional[int],
        parent: Optional["_MCTSNode"],
        prior: float = 0.25,
    ) -> None:
        self.action = action
        self.parent = parent
        self.children: Dict[int, "_MCTSNode"] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior = prior
        self.is_terminal: bool = False

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """PUCT score used by AlphaZero: Q(s,a) + U(s,a)."""
        q = self.mean_value
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + u

    def best_child(self, c_puct: float = 1.5) -> "_MCTSNode":
        return max(
            self.children.values(),
            key=lambda n: n.ucb_score(self.visit_count, c_puct),
        )

    def most_visited_child(self) -> "_MCTSNode":
        return max(self.children.values(), key=lambda n: n.visit_count)


def _clone_env(env):
    """Deep-copy the env's mutable state without cloning pygame assets."""
    from bobby_carrot.game import Bobby, MapInfo  # type: ignore

    new_env = object.__new__(type(env))
    new_env.__dict__.update(env.__dict__)

    # Deep copy only the mutable game-state fields
    old_info = env.map_info
    new_env.map_info = MapInfo(
        data=old_info.data.copy(),
        coord_start=old_info.coord_start,
        carrot_total=old_info.carrot_total,
        egg_total=old_info.egg_total,
    )

    old_bobby = env.bobby
    new_bobby = Bobby.__new__(Bobby)
    new_bobby.__dict__.update(old_bobby.__dict__)
    new_env.bobby = new_bobby

    # Shallow-copy deque (positions history)
    from collections import deque
    new_env.recent_positions = deque(env.recent_positions, maxlen=env.recent_positions.maxlen)

    # Shallow-copy sets
    new_env.target_positions = set(env.target_positions)
    new_env.finish_positions = set(env.finish_positions)

    # Reset render state so cloned env is always headless
    new_env._pygame = None
    new_env._screen = None
    new_env._render_assets = None
    new_env.headless = True

    return new_env


class MCTSEvaluator:
    """MCTS action selector using PPO policy/value as prior and leaf evaluator.

    At each decision step:
      1. Run n_sims simulations of depth ≤ max_depth from the current env state.
      2. Each simulation selects actions via PUCT (using PPO policy priors),
         rolls out until terminal or depth limit, then backs up the PPO value.
      3. Return the action with the highest visit count (most robust choice).

    Args:
        agent:        Trained PPOAgent (encoder + policy + value).
        preprocessor: ObservationPreprocessor matching the agent's input channels.
        n_sims:       Number of MCTS simulations per decision step.
        max_depth:    Maximum rollout depth per simulation.
        c_puct:       Exploration constant (higher = more exploration).
        device:       Torch device for the agent.
    """

    def __init__(
        self,
        agent: PPOAgent,
        preprocessor: ObservationPreprocessor,
        n_sims: int = 128,
        max_depth: int = 25,
        c_puct: float = 1.5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.agent = agent
        self.preprocessor = preprocessor
        self.n_sims = n_sims
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.device = device or next(agent.parameters()).device
        self.agent.eval()

    @torch.no_grad()
    def _get_policy_value(
        self, env, obs_raw: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Run one forward pass to get action priors and state value."""
        obs_t = self.preprocessor(obs_raw).unsqueeze(0)
        features = self.agent.encoder(obs_t)

        mask_np = env.get_valid_actions()
        mask_t = torch.tensor(mask_np, dtype=torch.bool, device=self.device).unsqueeze(0)

        dist = self.agent.policy(features, action_mask=mask_t)
        priors = dist.probs.squeeze(0).cpu().numpy()

        value = float(self.agent.value(features).item())
        return priors, value

    def _simulate(self, root_env, root_obs: np.ndarray) -> None:
        """Run one MCTS simulation from the root."""
        root = _MCTSNode(action=None, parent=None)

        # --- Selection + Expansion (single path from root) ---
        node = root
        env = _clone_env(root_env)
        obs = root_obs
        depth = 0
        cumulative_reward = 0.0
        discount = 1.0
        gamma = 0.99

        while depth < self.max_depth:
            if node.is_terminal:
                break

            priors, leaf_value = self._get_policy_value(env, obs)

            if not node.children:
                # Expand: create child nodes for all valid actions
                valid = env.get_valid_actions()
                for a in range(env.action_space_n):
                    if valid[a]:
                        node.children[a] = _MCTSNode(
                            action=a, parent=node, prior=float(priors[a])
                        )
                if not node.children:
                    node.is_terminal = True
                    break
                # Back up leaf value immediately on first expansion
                self._backup(node, leaf_value)
                return

            # Select child with highest PUCT score
            child = node.best_child(self.c_puct)
            obs_new, reward, done, _ = env.step(child.action)
            cumulative_reward += discount * reward
            discount *= gamma
            node = child
            obs = obs_new
            depth += 1

            if done:
                node.is_terminal = True
                # Terminal value = 0 (episode ended); cumulative reward already captured
                self._backup(node, 0.0)
                return

        # Reached depth limit — evaluate leaf with value network
        _, leaf_value = self._get_policy_value(env, obs)
        self._backup(node, leaf_value)

    def _backup(self, node: _MCTSNode, value: float) -> None:
        """Back-propagate value up to root."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent  # type: ignore[assignment]

    def select_action(self, env, obs_raw: np.ndarray) -> int:
        """Select the best action for the current env state via MCTS.

        Args:
            env:     The live BobbyCarrotEnv at the current decision point.
            obs_raw: Current raw observation array from env.reset() or env.step().

        Returns:
            Integer action index (0=LEFT, 1=RIGHT, 2=UP, 3=DOWN).
        """
        root = _MCTSNode(action=None, parent=None)

        # Run simulations
        for _ in range(self.n_sims):
            sim_env = _clone_env(env)
            sim_obs = obs_raw.copy()

            # ---- Selection ----
            node = root
            depth = 0
            done = False

            while node.children and not node.is_terminal and depth < self.max_depth:
                child = node.best_child(self.c_puct)
                sim_obs, reward, done, _ = sim_env.step(child.action)
                node = child
                depth += 1
                if done:
                    node.is_terminal = True
                    break

            if node.is_terminal or done:
                self._backup(node, 0.0)
                continue

            # ---- Expansion ----
            priors, leaf_value = self._get_policy_value(sim_env, sim_obs)
            valid = sim_env.get_valid_actions()
            expanded = False
            for a in range(sim_env.action_space_n):
                if valid[a]:
                    node.children[a] = _MCTSNode(
                        action=a, parent=node, prior=float(priors[a])
                    )
                    expanded = True

            if not expanded:
                node.is_terminal = True
                self._backup(node, 0.0)
                continue

            # ---- Rollout to depth limit ----
            rollout_node = random.choice(list(node.children.values()))
            sim_obs, reward, done, _ = sim_env.step(rollout_node.action)
            rollout_depth = 1

            while not done and rollout_depth < self.max_depth - depth:
                priors_r, _ = self._get_policy_value(sim_env, sim_obs)
                valid_r = sim_env.get_valid_actions()
                valid_actions = [a for a in range(sim_env.action_space_n) if valid_r[a]]
                if not valid_actions:
                    break
                # Sample action proportional to policy priors
                p = np.array([priors_r[a] for a in valid_actions], dtype=np.float64)
                p = p / p.sum()
                action = np.random.choice(valid_actions, p=p)
                sim_obs, reward, done, _ = sim_env.step(action)
                rollout_depth += 1

            # ---- Evaluation ----
            _, leaf_value = self._get_policy_value(sim_env, sim_obs)

            rollout_node.visit_count += 1
            rollout_node.total_value += leaf_value
            self._backup(node, leaf_value)

        if not root.children:
            # MCTS found nothing — fall back to greedy policy
            priors, _ = self._get_policy_value(env, obs_raw)
            valid = env.get_valid_actions()
            masked = priors * valid
            return int(np.argmax(masked))

        best = root.most_visited_child()
        return best.action  # type: ignore[return-value]
