"""Unified CLI entry point for training RL agents on Bobby Carrot.

Usage:
    python -m Bobby_Carrot.rl_models.train --algo ppo --timesteps 500000 --curriculum
    python -m Bobby_Carrot.rl_models.train --algo rainbow --timesteps 1000000
    python -m Bobby_Carrot.rl_models.train --algo ppo --icm --timesteps 500000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_GAME_PYTHON = _PROJECT_ROOT / "Game_Python"
if str(_GAME_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GAME_PYTHON))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Bobby_Carrot.rl_models.config import (
    PPOConfig,
    RainbowConfig,
    ICMConfig,
    TrainingConfig,
    LevelConfig,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train RL agents (PPO / Rainbow DQN) on Bobby Carrot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Algorithm selection
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "rainbow"],
                   help="Algorithm to train")
    p.add_argument("--icm", action="store_true", default=True, help="Enable ICM curiosity module")
    p.add_argument("--no-icm", action="store_false", dest="icm", help="Disable ICM curiosity module")

    # Training config
    p.add_argument("--timesteps", type=int, default=3_000_000, help="Total training timesteps")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    p.add_argument("--n-envs", type=int, default=4, help="Number of parallel envs (PPO)")
    p.add_argument("--max-steps", type=int, default=1500, help="Max steps per episode")

    # Curriculum
    p.add_argument("--curriculum", action="store_true", default=True, help="Enable curriculum learning")
    p.add_argument("--no-curriculum", action="store_false", dest="curriculum")
    p.add_argument("--reset-policy-head", action="store_true", default=True,
                   help="Reset policy head when resuming (for phase transfer)")
    p.add_argument("--no-reset-policy-head", action="store_false", dest="reset_policy_head")
    p.add_argument("--curriculum-start", type=int, default=5, help="Initial number of levels in curriculum (1-N)")
    p.add_argument("--curriculum-threshold", type=float, default=0.40, help="Success rate to promote (P4: lowered from 0.55)")

    # Logging
    p.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    p.add_argument("--log-interval", type=int, default=2000, help="Log every N timesteps")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    p.add_argument("--checkpoint-every", type=int, default=100_000, help="Checkpoint every N timesteps")
    p.add_argument("--eval-interval", type=int, default=50_000, help="Eval every N timesteps")

    # PPO-specific
    ppo_group = p.add_argument_group("PPO")
    ppo_group.add_argument("--ppo-lr", type=float, default=3e-4, help="PPO learning rate")
    ppo_group.add_argument("--ppo-clip", type=float, default=0.2, help="PPO clip ratio")
    ppo_group.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs")
    ppo_group.add_argument("--ppo-rollout", type=int, default=4096, help="PPO rollout length")
    ppo_group.add_argument("--ppo-minibatch", type=int, default=128, help="PPO minibatch size")
    ppo_group.add_argument("--ppo-entropy", type=float, default=0.15, help="Entropy coefficient (high for complex levels)")
    ppo_group.add_argument("--ppo-entropy-min", type=float, default=0.08, help="Minimum entropy coeff (P6: raised from 0.04)")
    ppo_group.add_argument("--ppo-gamma", type=float, default=0.99, help="Discount factor")
    ppo_group.add_argument("--ppo-gae-lambda", type=float, default=0.95, help="GAE lambda")

    # Rainbow-specific
    rb_group = p.add_argument_group("Rainbow DQN")
    rb_group.add_argument("--rainbow-lr", type=float, default=6.25e-5, help="Rainbow learning rate")
    rb_group.add_argument("--rainbow-batch", type=int, default=32, help="Rainbow batch size")
    rb_group.add_argument("--rainbow-buffer", type=int, default=200_000, help="Replay buffer size")
    rb_group.add_argument("--rainbow-n-step", type=int, default=5, help="N-step returns")
    rb_group.add_argument("--rainbow-atoms", type=int, default=51, help="C51 atom count")
    rb_group.add_argument("--rainbow-target-update", type=int, default=2000, help="Target net update freq")
    rb_group.add_argument("--rainbow-learning-starts", type=int, default=5000, help="Random steps before learning")

    # ICM-specific
    icm_group = p.add_argument_group("ICM")
    icm_group.add_argument("--icm-scale", type=float, default=0.01, help="Intrinsic reward scale")
    icm_group.add_argument("--icm-lr", type=float, default=1e-3, help="ICM learning rate")
    icm_group.add_argument("--icm-feature-dim", type=int, default=128, help="ICM feature dimension")

    # Level config
    p.add_argument("--train-normal-max", type=int, default=25,
                   help="Train on normal levels 1..N")
    p.add_argument("--train-egg-max", type=int, default=0,
                   help="Train on egg levels 1..N (0 to disable)")
    p.add_argument("--test-normal-start", type=int, default=26,
                   help="Test normal levels start")
    p.add_argument("--test-normal-end", type=int, default=30,
                   help="Test normal levels end")
    p.add_argument("--test-egg-start", type=int, default=0,
                   help="Test egg levels start (0 to disable)")
    p.add_argument("--test-egg-end", type=int, default=0,
                   help="Test egg levels end (0 to disable)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Build configs from CLI args
    train_levels = [("normal", i) for i in range(1, args.train_normal_max + 1)]
    if args.train_egg_max > 0:
        train_levels += [("egg", i) for i in range(1, args.train_egg_max + 1)]

    test_levels = []
    if args.test_normal_start > 0 and args.test_normal_end >= args.test_normal_start:
        test_levels += [("normal", i) for i in range(args.test_normal_start, args.test_normal_end + 1)]
    if args.test_egg_start > 0 and args.test_egg_end >= args.test_egg_start:
        test_levels += [("egg", i) for i in range(args.test_egg_start, args.test_egg_end + 1)]

    level_config = LevelConfig(
        train_levels=train_levels,
        test_levels=test_levels,
    )

    train_config = TrainingConfig(
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=Path(args.checkpoint_dir),
        checkpoint_every=args.checkpoint_every,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        log_dir=Path(args.log_dir),
        curriculum=args.curriculum,
        curriculum_start_levels=args.curriculum_start,
        curriculum_promotion_threshold=args.curriculum_threshold,
        max_steps_per_episode=args.max_steps,
        n_envs=args.n_envs,
        reset_policy_head_on_resume=args.reset_policy_head,
    )

    icm_config = ICMConfig(
        enabled=args.icm,
        lr=args.icm_lr,
        feature_dim=args.icm_feature_dim,
        intrinsic_reward_scale=args.icm_scale,
    )

    print("=" * 70)
    print(f"  Bobby Carrot RL Training - {args.algo.upper()}"
          + (" + ICM" if args.icm else ""))
    print("=" * 70)
    print(f"  Timesteps:     {args.timesteps:,}")
    print(f"  Curriculum:    {'ON' if args.curriculum else 'OFF'}")
    print(f"  Train levels:  {len(level_config.train_levels)}")
    print(f"  Test levels:   {len(level_config.test_levels)}")
    print(f"  Device:        {args.device}")
    print("=" * 70)

    if args.algo == "ppo":
        ppo_config = PPOConfig(
            lr=args.ppo_lr,
            clip_ratio=args.ppo_clip,
            n_epochs=args.ppo_epochs,
            rollout_length=args.ppo_rollout,
            minibatch_size=args.ppo_minibatch,
            entropy_coeff=args.ppo_entropy,
            entropy_min=args.ppo_entropy_min,
            gamma=args.ppo_gamma,
            gae_lambda=args.ppo_gae_lambda,
        )
        from Bobby_Carrot.rl_models.ppo import train_ppo
        train_ppo(ppo_config, train_config, level_config, icm_config)

    elif args.algo == "rainbow":
        rainbow_config = RainbowConfig(
            lr=args.rainbow_lr,
            batch_size=args.rainbow_batch,
            buffer_size=args.rainbow_buffer,
            n_step=args.rainbow_n_step,
            atom_size=args.rainbow_atoms,
            target_update_freq=args.rainbow_target_update,
            learning_starts=args.rainbow_learning_starts,
        )
        from Bobby_Carrot.rl_models.rainbow import train_rainbow
        train_rainbow(rainbow_config, train_config, level_config, icm_config)


if __name__ == "__main__":
    main()
