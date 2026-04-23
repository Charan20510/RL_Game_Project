"""Intrinsic Curiosity Module (ICM) — composable add-on for exploration.

Provides intrinsic reward based on prediction error of a forward dynamics model.
Can be attached to either PPO or Rainbow DQN.

Architecture:
    - Forward Model: predicts φ(s') from (φ(s), a)
    - Inverse Model: predicts action a from (φ(s), φ(s'))
    - Intrinsic Reward: ||φ_predicted(s') - φ_actual(s')||²

The feature encoder φ is the CNN encoder from the main agent (passed in
as pre-computed features). ICM adds its own projection layer for a
compact feature space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ICMConfig


class ICMModule(nn.Module):
    """Intrinsic Curiosity Module.

    Works with pre-computed encoder features from the main agent.
    This keeps ICM decoupled from the specific agent architecture.
    """

    def __init__(
        self,
        config: ICMConfig,
        encoder_dim: int,
        n_actions: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_actions = n_actions
        self.feature_dim = config.feature_dim

        # Feature projection: encoder output → compact ICM feature space
        self.feature_proj = nn.Sequential(
            nn.Linear(encoder_dim, config.feature_dim),
            nn.ReLU(inplace=True),
        )

        # Forward model: predicts φ(s') from (φ(s), one-hot(a))
        self.forward_model = nn.Sequential(
            nn.Linear(config.feature_dim + n_actions, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, config.feature_dim),
        )

        # Inverse model: predicts action from (φ(s), φ(s'))
        self.inverse_model = nn.Sequential(
            nn.Linear(config.feature_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )

        # Running mean/var for reward normalisation
        self.reward_running_mean = 0.0
        self.reward_running_var = 1.0
        self._reward_count = 0

    def _project_features(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Project raw encoder features to ICM feature space."""
        return self.feature_proj(encoder_features)

    def intrinsic_reward(
        self,
        enc_obs: torch.Tensor,
        enc_next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> float:
        """Compute intrinsic reward for a single transition.

        Args:
            enc_obs: encoder features for current state (1, encoder_dim)
            enc_next_obs: encoder features for next state (1, encoder_dim)
            actions: action taken (1,) long tensor

        Returns:
            Normalised intrinsic reward (float).
        """
        with torch.no_grad():
            phi_s = self._project_features(enc_obs)
            phi_s_next = self._project_features(enc_next_obs)

            # One-hot encode action
            action_onehot = F.one_hot(actions, self.n_actions).float()
            forward_input = torch.cat([phi_s, action_onehot], dim=-1)

            phi_s_next_pred = self.forward_model(forward_input)

            # Prediction error as intrinsic reward
            error = F.mse_loss(phi_s_next_pred, phi_s_next, reduction="none").sum(dim=-1)
            reward = error.item()

            # Update running statistics for normalisation
            self._reward_count += 1
            decay = self.config.reward_running_mean_decay
            self.reward_running_mean = decay * self.reward_running_mean + (1 - decay) * reward
            self.reward_running_var = decay * self.reward_running_var + (1 - decay) * (reward - self.reward_running_mean) ** 2

            # Normalise
            std = max(self.reward_running_var ** 0.5, 1e-8)
            normalised_reward = reward / std

        return normalised_reward

    def compute_loss(
        self,
        enc_obs: torch.Tensor,
        enc_next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined forward + inverse model loss for training.

        Args:
            enc_obs: (B, encoder_dim) detached encoder features
            enc_next_obs: (B, encoder_dim) detached encoder features
            actions: (B,) long tensor

        Returns:
            Scalar loss tensor.
        """
        phi_s = self._project_features(enc_obs)
        phi_s_next = self._project_features(enc_next_obs)

        # Forward loss: predict φ(s') from (φ(s), a)
        action_onehot = F.one_hot(actions, self.n_actions).float()
        forward_input = torch.cat([phi_s, action_onehot], dim=-1)
        phi_s_next_pred = self.forward_model(forward_input)
        forward_loss = F.mse_loss(phi_s_next_pred, phi_s_next.detach())

        # Inverse loss: predict action from (φ(s), φ(s'))
        inverse_input = torch.cat([phi_s, phi_s_next], dim=-1)
        action_logits = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(action_logits, actions)

        # Combined loss
        total_loss = (
            self.config.forward_loss_weight * forward_loss
            + self.config.inverse_loss_weight * inverse_loss
        )

        return total_loss
