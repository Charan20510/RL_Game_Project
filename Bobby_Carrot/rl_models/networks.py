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
        return self.process_numpy_batch(np.asarray(obs, dtype=np.int16)[None])[0]

    @staticmethod
    def _parse_obs_batch(obs_array: np.ndarray):
        """Parse raw obs batch → (px, py, inv, tiles_hw, path_hw, finish_hw).

        tiles_hw / path_hw / finish_hw have shape (B, 16, 16).
        tiles_hw[b, y, x] == tiles[b, x + y*16] because reshape is row-major
        and tiles[x + y*16] == tiles[y*16 + x].
        """
        B, obs_len = obs_array.shape
        px = obs_array[:, 0].astype(np.int32)
        py = obs_array[:, 1].astype(np.int32)

        path  = np.zeros((B, 256), dtype=np.int16)
        finish = np.zeros((B, 256), dtype=np.int16)

        if obs_len >= 2 + 4 + 256 + 256 + 256:
            inv    = obs_array[:, 2:6]
            tiles  = obs_array[:, 6:262]
            path   = obs_array[:, 262:518]
            finish = obs_array[:, 518:774]
        elif obs_len >= 2 + 4 + 256 + 256:
            inv   = obs_array[:, 2:6]
            tiles = obs_array[:, 6:262]
            path  = obs_array[:, 262:518]
        elif obs_len >= 2 + 4 + 256:
            inv   = obs_array[:, 2:6]
            tiles = obs_array[:, 6:262]
        elif obs_len >= 2 + 256:
            inv   = np.zeros((B, 4), dtype=np.int16)
            tiles = obs_array[:, 2:258]
        else:
            inv   = np.zeros((B, 4), dtype=np.int16)
            tiles = np.zeros((B, 256), dtype=np.int16)

        tiles_hw  = tiles.reshape(B, 16, 16)
        path_hw   = path.reshape(B, 16, 16)
        finish_hw = finish.reshape(B, 16, 16)
        return px, py, inv, tiles_hw, path_hw, finish_hw

    def process_numpy_batch(self, obs_array: np.ndarray) -> torch.Tensor:
        """Vectorised batch: (B, obs_dim) int16 → (B, C, 16, 16) float32 on device."""
        obs_array = np.asarray(obs_array, dtype=np.int16)
        B = obs_array.shape[0]
        px, py, inv, tw, ph, fh = self._parse_obs_batch(obs_array)

        ch = np.zeros((B, _NUM_OBS_CHANNELS, 16, 16), dtype=np.float32)

        # Tile channels (vectorised over full batch)
        ch[:, 0]  = (tw >= 18)
        ch[:, 1]  = (tw == 19)
        ch[:, 2]  = (tw == 45)
        ch[:, 3]  = (tw == 44)
        ch[:, 4]  = (tw == 30)
        ch[:, 5]  = (tw == 31) | (tw == 46)
        ch[:, 6]  = np.isin(tw, [32, 34, 36])
        ch[:, 7]  = np.isin(tw, [33, 35, 37])
        ch[:, 10] = (tw == 40)
        ch[:, 11] = (tw == 41)
        ch[:, 12] = (tw == 42)
        ch[:, 13] = (tw == 43)
        ch[:, 14] = np.isin(tw, [24, 25, 26, 27, 28, 29])
        ch[:, 15] = np.isin(tw, [22, 23, 38, 39])
        ch[:, 16] = (tw == 20)

        # Agent position (one cell per sample)
        valid = (px >= 0) & (px < 16) & (py >= 0) & (py < 16)
        b_idx = np.where(valid)[0]
        ch[b_idx, 8, py[b_idx], px[b_idx]] = 1.0

        # Inventory channel
        key_gray   = inv[:, 0].astype(np.float32)
        key_yellow = inv[:, 1].astype(np.float32)
        key_red    = inv[:, 2].astype(np.float32)
        remaining  = inv[:, 3].astype(np.float32) / 10.0
        key_val = (key_gray * 0.33 + key_yellow * 0.33 + key_red * 0.34)[:, None, None]
        ch[:, 9, 0:8, 0:8]   = key_val
        ch[:, 9, 8:16, 8:16] = remaining[:, None, None]

        # Path trace and finish-critical path
        ch[:, 17] = ph.astype(np.float32) / 32.0
        ch[:, 18] = fh.astype(np.float32)

        return torch.from_numpy(ch).to(self.device)

    def process_batch(self, obs_list: List[np.ndarray]) -> torch.Tensor:
        """Process a list of observations → (B, C, 16, 16)."""
        return self.process_numpy_batch(np.stack(obs_list))



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
