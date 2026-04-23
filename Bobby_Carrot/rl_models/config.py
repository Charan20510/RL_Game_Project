"""Unified configuration dataclasses for all RL algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class LevelConfig:
    """Defines which levels are used for training and testing."""

    train_levels: List[Tuple[str, int]] = field(default_factory=lambda: (
        [("normal", i) for i in range(1, 26)]   # normal 1-25
    ))
    test_levels: List[Tuple[str, int]] = field(default_factory=lambda: (
        [("normal", i) for i in range(26, 31)]  # normal 26-30
    ))


@dataclass
class TrainingConfig:
    """Shared training orchestration settings."""

    total_timesteps: int = 3_000_000     # 25 levels need ~120k steps each
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every: int = 100_000
    eval_interval: int = 50_000
    eval_episodes_per_level: int = 20
    log_interval: int = 2_000
    log_dir: Path = Path("logs")

    # Curriculum settings
    curriculum: bool = True
    curriculum_start_levels: int = 1      # Start with L1 only; promote as mastery accrues
    curriculum_promotion_window: int = 120
    curriculum_promotion_threshold: float = 0.35  # Lowered 0.40→0.35: Phase-1 L2 peaked at 57% but oscillated; 0.35 matches fallback threshold and unblocks curriculum flow
    curriculum_add_levels: int = 3        # Add 3 levels per promotion
    # Anti-forgetting: keep mastered levels practiced even when new levels are
    # failing hard, and require 2 consecutive windows above threshold before
    # admitting the next batch of levels.
    level_history_window: int = 80
    curriculum_mastery_floor: float = 0.55  # Min sampling weight for a mastered (≥75%) level
    curriculum_min_quota: float = 0.10    # Every active level must get ≥ this fraction (10% with 25 levels)
    curriculum_max_level_weight: float = 0.40  # No level may consume more than this fraction of samples
    curriculum_dwell_windows: int = 2     # Highest level must stay ≥ threshold this many windows
    # Fallback promotion: if the highest level has plateaued above this threshold
    # for `fallback_windows` consecutive windows without reaching the main
    # promotion threshold, promote anyway so later levels still get training
    # exposure. Threshold matches main promotion to avoid soft-path advancement.
    curriculum_fallback_threshold: float = 0.35
    curriculum_fallback_windows: int = 4  # Lowered 6→4: prior 500k run never triggered fallback on oscillating L2; 4-window sustain is enough evidence
    # On each promotion, boost entropy_coeff for N steps to force exploration
    # on the newly-added levels before the LR schedule tightens it.
    entropy_boost_steps: int = 30_000  # Lowered 80k→30k: prior run spent ~40% of total timesteps under entropy boost due to spurious regression re-arms
    entropy_boost_multiplier: float = 2.0
    # P6 regression-triggered entropy boost: if any active level's rolling
    # success drops by >= this from its recorded max, re-arm the entropy boost.
    regression_trigger_drop: float = 0.40  # Raised 0.25→0.40: 0.25 fired on normal L2 variance (57%→32% swings) causing entropy-boost cascade that prevented policy consolidation
    # P5 EMA teacher + KL anti-forgetting.
    # Decay is applied ONCE PER ROLLOUT (not per minibatch). At 0.99/rollout
    # the teacher absorbs ~1% of the student's update each rollout, giving a
    # stable ~50-rollout memory window. The old 0.995/minibatch value caused
    # 47% teacher update per rollout (128 steps × 0.5%), losing L_n-1 knowledge
    # within 2-3 rollouts of a new level being introduced.
    teacher_ema_decay: float = 0.99        # Per-rollout decay (NOT per-minibatch)
    teacher_kl_coef: float = 0.02          # Global KL(student || teacher) weight
    teacher_kl_mastery_coef: float = 0.20  # Raised from 0.10: L1 was forgotten every time L2/L3 gradients dominated a rollout
    # Retention gate: block curriculum promotion if any already-active mastered
    # level (peak success ≥70%) has regressed below this floor. 0.0 disables.
    curriculum_retention_floor: float = 0.60
    # Cosine decay of LR over the last fraction of training (0.0 disables).
    lr_decay_final_fraction: float = 0.25  # Decay over last 25%
    lr_decay_min_multiplier: float = 0.3   # LR floor = lr * 0.3

    # Observation
    observation_mode: str = "full"
    max_steps_per_episode: int = 1500     # Level 26 has 65 carrots — needs headroom
    reward_scale: float = 1.0

    # Adaptive exploration: force random actions on levels the agent hasn't learned
    exploration_epsilon: float = 0.25
    exploration_success_threshold: float = 0.3

    # Multi-env (vectorized)
    n_envs: int = 1  # Single env for simplicity

    # Transfer learning: reset policy head when resuming from a different phase.
    # Default False — action semantics (L/R/U/D) are identical across all Bobby
    # Carrot levels, so discarding learned action preferences is almost always
    # harmful. Only set True if the action space *semantics* change.
    reset_policy_head_on_resume: bool = False

    # Early stopping (single-level demo runs). 0.0 disables — every existing
    # phased run passes the default, so this is a no-op for them.
    early_stop_success: float = 0.0       # Stop when rolling success >= this
    early_stop_window: int = 100          # Over the last N episodes
    early_stop_min_timesteps: int = 20_000  # Floor before checks arm

    # Greedy-stability gate (single-level reliability runs).
    # When enabled, rollout-based "best" snapshots are disabled and progress is
    # judged from periodic greedy evaluation windows.
    greedy_gate_enabled: bool = False
    greedy_gate_threshold: float = 0.95
    greedy_gate_required_windows: int = 10


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters."""

    lr: float = 1e-4           # Full LR — cold start on new 25-level split
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2    # Default PPO clip
    value_coeff: float = 0.5
    entropy_coeff: float = 0.15  # High start — exploring 25 levels from scratch
    max_grad_norm: float = 0.5
    rollout_length: int = 8192   # Larger rollout covers more level variety per update
    n_epochs: int = 3
    minibatch_size: int = 128    # Scaled with rollout length
    normalize_advantages: bool = True
    entropy_min: float = 0.10    # P6: raised from 0.04 — late levels need sustained exploration

    # Network architecture
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 64])
    hidden_dim: int = 256


@dataclass
class RainbowConfig:
    """Rainbow DQN hyperparameters (all 6 enhancements)."""

    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64         # 64 gives more stable distributional updates than 32
    buffer_size: int = 100_000
    learning_starts: int = 10_000  # Fill buffer with diverse experiences before first update
    target_update_freq: int = 1_000
    max_grad_norm: float = 10.0

    # Double DQN — enabled by default (no extra param needed)

    # Dueling
    hidden_dim: int = 256

    # PER (Prioritized Experience Replay)
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 200_000
    per_epsilon: float = 1e-6

    # NoisyNet
    noisy_std: float = 0.5

    # N-step returns
    n_step: int = 3              # 3-step keeps targets stable under sparse rewards

    # C51 (Distributional)
    # v_min/v_max must bracket the actual cumulative return range.
    # With finish=300, 13 carrots×15×1.5≈293, approach bonuses ~100 → max ~700.
    # Minimum: death(-50) + step penalties + crumble penalties → min ~-300.
    atom_size: int = 51
    v_min: float = -300.0
    v_max: float = 700.0

    # Network architecture
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 64])


@dataclass
class ICMConfig:
    """Intrinsic Curiosity Module hyperparameters."""

    enabled: bool = True   # Enabled by default for crumble-heavy levels
    lr: float = 1e-3
    feature_dim: int = 128
    intrinsic_reward_scale: float = 0.02
    forward_loss_weight: float = 0.2
    inverse_loss_weight: float = 0.8
    reward_running_mean_decay: float = 0.99
