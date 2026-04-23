"""Neural network components shared across all RL algorithms.

Includes:
- ObservationPreprocessor: converts raw env observations to multi-channel tensors
- CNNEncoder: shared convolutional backbone
- PolicyHead / ValueHead: for PPO (with action masking support)
- NoisyLinear: factorised Gaussian noise layer for NoisyNet
- DuelingDistributionalHead: combined Dueling + C51 head for Rainbow
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Observation Preprocessing
# ---------------------------------------------------------------------------

# Tile category mapping (must match rl_env.py tile IDs)
_NUM_OBS_CHANNELS = 19  # 17 semantic tile channels + path trace + finish-critical path


class ObservationPreprocessor:
    """Converts raw 16x16 tile-grid observations into multi-channel float tensors.

    Channel layout (19 channels total):
        0:  Walkable ground       (tile >= 18)
        1:  Carrot                (tile == 19)
        2:  Egg                   (tile == 45)
        3:  Finish tile           (tile == 44)
        4:  Active crumble        (tile == 30)
        5:  Hazard / collapsed    (tile == 31 or 46)
        6:  Key pickup            (tile in {32, 34, 36})
        7:  Door locked           (tile in {33, 35, 37})
        8:  Agent position        (1 at agent pos)
        9:  Inventory info        (remaining targets normalised + key flags)
        10: Conveyor forces LEFT  (tile == 40)   ← directional semantics split
        11: Conveyor forces RIGHT (tile == 41)
        12: Conveyor forces UP    (tile == 42)
        13: Conveyor forces DOWN  (tile == 43)
        14: Arrow / bi-dir conv   (tiles 24-29)  ← restricts entry direction
        15: Switch                (tiles 22, 23, 38, 39)  ← global state flip
        16: Visited-safe          (tile == 20, previously-a-carrot, now walkable)
        17: Path trace history    (recent positions visited)
        18: Finish-critical path  (BFS shortest path from agent to finish tile)
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @staticmethod
    def num_channels() -> int:
        return _NUM_OBS_CHANNELS

    def __call__(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a single observation array to (C, 16, 16) float tensor."""
        return self.process_single(obs)

    def process_single(self, obs: np.ndarray) -> torch.Tensor:
        """Process one observation → (C, 16, 16)."""
        obs = np.asarray(obs, dtype=np.int16)
        channels = np.zeros((_NUM_OBS_CHANNELS, 16, 16), dtype=np.float32)

        # First 2 values are always (px, py)
        px, py = int(obs[0]), int(obs[1])

        # Determine inventory offset
        # Full mode: [px, py, <inv 4>, <tiles 256>, <path 256 (optional)>]
        obs_len = len(obs)
        path_grid = np.zeros(256, dtype=np.int16)
        finish_path_grid = np.zeros(256, dtype=np.int16)

        if obs_len >= 2 + 4 + 256 + 256 + 256:
            # With inventory + path trace + finish-critical path
            inv = obs[2:6]
            tiles = obs[6:6 + 256]
            path_grid = obs[6 + 256 : 6 + 512]
            finish_path_grid = obs[6 + 512 : 6 + 768]
        elif obs_len >= 2 + 4 + 256 + 256:
            # With inventory + path trace (no finish path)
            inv = obs[2:6]
            tiles = obs[6:6 + 256]
            path_grid = obs[6 + 256 : 6 + 512]
        elif obs_len >= 2 + 256 + 256:
            # Without inventory + path trace
            inv = np.zeros(4, dtype=np.int16)
            tiles = obs[2:2 + 256]
            path_grid = obs[2 + 256 : 2 + 512]
        elif obs_len >= 2 + 4 + 256:
            # With inventory (no path)
            inv = obs[2:6]
            tiles = obs[6:6 + 256]
        elif obs_len >= 2 + 256:
            # Without inventory (no path)
            inv = np.zeros(4, dtype=np.int16)
            tiles = obs[2:2 + 256]
        else:
            # Local/compact mode — create a blank 16×16 and fill what we can
            inv = np.zeros(4, dtype=np.int16)
            tiles = np.zeros(256, dtype=np.int16)
            remaining = obs[2:] if obs_len > 2 else np.array([], dtype=np.int16)
            local_size = int(math.isqrt(len(remaining))) if len(remaining) > 0 else 0
            if local_size > 0:
                half = local_size // 2
                for idx, val in enumerate(remaining):
                    dy = idx // local_size - half
                    dx = idx % local_size - half
                    gx, gy = px + dx, py + dy
                    if 0 <= gx < 16 and 0 <= gy < 16:
                        tiles[gx + gy * 16] = val

        # Fill tile channels
        for y in range(16):
            for x in range(16):
                tile = int(tiles[x + y * 16])
                if tile >= 18:
                    channels[0, y, x] = 1.0
                if tile == 19:
                    channels[1, y, x] = 1.0
                if tile == 45:
                    channels[2, y, x] = 1.0
                if tile == 44:
                    channels[3, y, x] = 1.0
                if tile == 30:
                    channels[4, y, x] = 1.0
                if tile == 31 or tile == 46:
                    channels[5, y, x] = 1.0
                if tile in (32, 34, 36):
                    channels[6, y, x] = 1.0
                if tile in (33, 35, 37):
                    channels[7, y, x] = 1.0
                # Directional conveyor channels (10-13) — each direction is a
                # separate semantic plane so the CNN can learn "this tile forces
                # LEFT" independently from "this tile forces RIGHT".
                if tile == 40:
                    channels[10, y, x] = 1.0   # forces LEFT
                if tile == 41:
                    channels[11, y, x] = 1.0   # forces RIGHT
                if tile == 42:
                    channels[12, y, x] = 1.0   # forces UP
                if tile == 43:
                    channels[13, y, x] = 1.0   # forces DOWN
                # Arrow & bi-directional conveyor tiles (24-29): restrict which
                # directions are legal to enter/exit — grouped as one channel
                # since the shared semantics are "directional restriction".
                if tile in (24, 25, 26, 27, 28, 29):
                    channels[14, y, x] = 1.0
                # Switch tiles (22, 23, 38, 39): stepping triggers a global
                # map-state flip — distinct from directional mechanics.
                if tile in (22, 23, 38, 39):
                    channels[15, y, x] = 1.0
                # Visited-safe: tile==20 means a carrot was collected here.
                # tile==46 is already on channel 5 (hazard) — no duplication.
                if tile == 20:
                    channels[16, y, x] = 1.0

        # Agent position channel
        if 0 <= px < 16 and 0 <= py < 16:
            channels[8, py, px] = 1.0

        # Inventory channel — broadcast key/remaining info
        key_gray = float(inv[0]) if len(inv) > 0 else 0.0
        key_yellow = float(inv[1]) if len(inv) > 1 else 0.0
        key_red = float(inv[2]) if len(inv) > 2 else 0.0
        remaining = float(inv[3]) / 10.0 if len(inv) > 3 else 0.0
        # Encode as a spatial pattern: top-left quadrant = keys, bottom-right = remaining
        channels[9, 0:8, 0:8] = key_gray * 0.33 + key_yellow * 0.33 + key_red * 0.34
        channels[9, 8:16, 8:16] = remaining

        # Channel 17: Path Trace History (visit counts, normalised by loop_window=32)
        # Values are 0..32 integers from the env; dividing by 32 gives a [0,1] signal
        # that encodes how recently/frequently the agent has occupied each cell.
        # High values mark oscillation hot-spots; the policy can learn to avoid them.
        channels[17, :, :] = path_grid.reshape((16, 16)).astype(np.float32) / 32.0

        # Channel 18: Finish-Critical Path (BFS shortest path to finish)
        channels[18, :, :] = finish_path_grid.reshape((16, 16)).astype(np.float32)

        return torch.from_numpy(channels).to(self.device)

    def process_batch(self, obs_list: List[np.ndarray]) -> torch.Tensor:
        """Process a list of observations → (B, C, 16, 16)."""
        tensors = [self.process_single(o) for o in obs_list]
        return torch.stack(tensors)

    def process_numpy_batch(self, obs_array: np.ndarray) -> torch.Tensor:
        """Process a numpy array of shape (B, obs_dim) → (B, C, 16, 16)."""
        return self.process_batch([obs_array[i] for i in range(obs_array.shape[0])])


# ---------------------------------------------------------------------------
# CNN Encoder (shared backbone)
# ---------------------------------------------------------------------------

def init_orthogonal(module: nn.Module, gain: float = 1.0) -> None:
    """Initialize weights orthogonally and biases to zero."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CNNEncoder(nn.Module):
    """Shared CNN backbone for processing 16x16 multi-channel grid observations.

    Architecture (uses GroupNorm instead of BatchNorm for RL stability):
        Conv2d(in, 32, 3, pad=1) → GN → ReLU
        Conv2d(32, 64, 3, pad=1) → GN → ReLU
        Conv2d(64, 64, 3, pad=1) → GN → ReLU
        Conv2d(64, 64, 3, stride=2) → GN → ReLU
        Flatten → Linear → ReLU → Linear → ReLU

    Note: GroupNorm(1, C) is equivalent to LayerNorm and normalizes
    per-sample. BatchNorm is harmful in RL because running statistics
    get corrupted across different levels/episodes.
    """

    def __init__(
        self,
        in_channels: int = _NUM_OBS_CHANNELS,
        channel_sizes: List[int] | None = None,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if channel_sizes is None:
            channel_sizes = [32, 64, 64, 64]

        layers: list[nn.Module] = []
        prev_ch = in_channels
        for i, ch in enumerate(channel_sizes):
            stride = 2 if i == len(channel_sizes) - 1 else 1
            layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=3, stride=stride, padding=1),
                nn.GroupNorm(1, ch),  # Per-sample norm (LayerNorm equivalent)
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch

        self.conv = nn.Sequential(*layers)

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 16, 16)
            conv_out = self.conv(dummy)
            self._conv_flat_size = int(conv_out.view(1, -1).shape[1])

        self.fc = nn.Sequential(
            nn.Linear(self._conv_flat_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = hidden_dim

        # Apply orthogonal initialization to all conv/linear layers
        self.apply(lambda m: init_orthogonal(m, gain=nn.init.calculate_gain('relu')))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x shape: (B, C, 16, 16) → (B, hidden_dim)."""
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


# ---------------------------------------------------------------------------
# PPO Network Heads
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """Categorical policy head for discrete action space with action masking."""

    def __init__(self, input_dim: int, n_actions: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, n_actions)
        # Small gain ensures initial policy is uniformly random
        init_orthogonal(self.linear, gain=0.01)

    def forward(
        self,
        features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Categorical:
        """Compute action distribution, optionally masking invalid actions.

        Args:
            features: (B, input_dim) encoder output.
            action_mask: (B, n_actions) bool tensor. True = valid, False = masked.
                         If None, all actions are considered valid.
        """
        logits = self.linear(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        # Replace any NaN/Inf (e.g. from NaN encoder features during a bad update)
        # with 0 so Categorical always gets finite logits.  A NaN-replaced row
        # produces a near-uniform distribution for that step, which is safe.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
        return torch.distributions.Categorical(logits=logits, validate_args=False)


class ValueHead(nn.Module):
    """Scalar value head for V(s)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        init_orthogonal(self.linear, gain=1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


# ---------------------------------------------------------------------------
# NoisyNet Linear Layer
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Factorised NoisyNet linear layer.

    Replaces standard nn.Linear with learnable noise parameters for
    exploration without ε-greedy. Uses factorised Gaussian noise for
    memory efficiency.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorised noise — not learnable, registered as buffers
        self.weight_epsilon: torch.Tensor
        self.bias_epsilon: torch.Tensor
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Dueling + Distributional (C51) Head for Rainbow DQN
# ---------------------------------------------------------------------------

class DuelingDistributionalHead(nn.Module):
    """Combined Dueling + C51 distributional head.

    Outputs per-action categorical distributions over atoms.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    where V and A are distributional (each outputs atom_size logits).
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int = 4,
        atom_size: int = 51,
        v_min: float = -100.0,
        v_max: float = 200.0,
        hidden_dim: int = 256,
        noisy_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max

        # Support atoms
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, atom_size),
        )

        # Value stream (Noisy)
        self.v_fc = NoisyLinear(input_dim, hidden_dim, noisy_std)
        self.v_out = NoisyLinear(hidden_dim, atom_size, noisy_std)

        # Advantage stream (Noisy)
        self.a_fc = NoisyLinear(input_dim, hidden_dim, noisy_std)
        self.a_out = NoisyLinear(hidden_dim, n_actions * atom_size, noisy_std)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities over atoms for each action.

        Output shape: (B, n_actions, atom_size)
        """
        batch_size = features.size(0)

        # Value stream
        v = F.relu(self.v_fc(features))
        v = self.v_out(v).view(batch_size, 1, self.atom_size)

        # Advantage stream
        a = F.relu(self.a_fc(features))
        a = self.a_out(a).view(batch_size, self.n_actions, self.atom_size)

        # Dueling combination
        q_atoms = v + a - a.mean(dim=1, keepdim=True)

        # Log-softmax over atoms
        log_probs = F.log_softmax(q_atoms, dim=-1)
        return log_probs

    def q_values(self, features: torch.Tensor) -> torch.Tensor:
        """Compute expected Q-values from the distributional output.

        Returns shape: (B, n_actions)
        """
        log_probs = self.forward(features)
        probs = log_probs.exp()
        return (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    def reset_noise(self) -> None:
        self.v_fc.reset_noise()
        self.v_out.reset_noise()
        self.a_fc.reset_noise()
        self.a_out.reset_noise()
