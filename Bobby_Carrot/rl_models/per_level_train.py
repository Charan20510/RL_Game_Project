"""Per-level PPO training with individual checkpoints.

Trains a PPO agent on each level independently.  After training completes
for a level, a checkpoint is saved to:
    <ckpt_dir>/normal_<NN>_final.pt

Each level starts from either:
  - A BC pre-trained checkpoint (recommended) via --bc-dir
  - The previous level's final checkpoint (sequential transfer) via --sequential
  - Random initialization (no warm-start)

Usage:
    # Full pipeline: BC warm-start, per-level PPO, 200k steps/level
    python -m Bobby_Carrot.rl_models.per_level_train \\
        --bc-dir checkpoints/bc \\
        --ckpt-dir checkpoints/per_level \\
        --timesteps 200000

    # Sequential transfer (each level continues from the previous)
    python -m Bobby_Carrot.rl_models.per_level_train \\
        --sequential --ckpt-dir checkpoints/per_level

    # Single level (useful for testing)
    python -m Bobby_Carrot.rl_models.per_level_train \\
        --levels 5 --bc-dir checkpoints/bc --ckpt-dir checkpoints/per_level
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent.parent
_GAME_PYTHON = _HERE / "Game_Python"
if str(_GAME_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GAME_PYTHON))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bobby_carrot.rl_env import BobbyCarrotEnv  # type: ignore
from Bobby_Carrot.rl_models.config import PPOConfig, TrainingConfig
from Bobby_Carrot.rl_models.networks import ObservationPreprocessor
from Bobby_Carrot.rl_models.ppo import PPOAgent, RunningMeanStd
from Bobby_Carrot.rl_models.buffers import RolloutBuffer


# Disable distribution argument validation globally (same as ppo.py)
torch.distributions.Distribution.set_default_validate_args(False)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LevelTrainResult:
    map_kind: str
    map_number: int
    timesteps: int
    episodes: int
    best_success: float
    final_success: float
    checkpoint_path: Path
    elapsed_sec: float


# ---------------------------------------------------------------------------
# Core per-level PPO trainer
# ---------------------------------------------------------------------------

def train_ppo_single_level(
    map_kind: str,
    map_number: int,
    ckpt_dir: Path,
    total_timesteps: int = 200_000,
    resume_from: Optional[Path] = None,
    ppo_config: Optional[PPOConfig] = None,
    max_steps_per_episode: int = 1500,
    observation_mode: str = "full",
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    verbose: bool = True,
    log_interval: int = 5_000,
    teacher_kl_coef: float = 0.05,
    explore_eps: float = 0.10,
) -> LevelTrainResult:
    """Run PPO on a single level and save a checkpoint on completion.

    This is a lightweight trainer — no curriculum, no ICM, no teacher —
    designed for focused per-level training after BC pre-training.
    """
    level_tag = f"{map_kind}_{map_number:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = ckpt_dir / f"{level_tag}_final.pt"

    # Already trained?
    if final_ckpt_path.exists():
        if verbose:
            print(f"  [{level_tag}] Checkpoint exists, skipping. ({final_ckpt_path})")
        ckpt = torch.load(str(final_ckpt_path), map_location=device, weights_only=False)
        return LevelTrainResult(
            map_kind=map_kind,
            map_number=map_number,
            timesteps=ckpt.get("total_timesteps", 0),
            episodes=ckpt.get("episode_count", 0),
            best_success=ckpt.get("best_success", 0.0),
            final_success=ckpt.get("final_success", 0.0),
            checkpoint_path=final_ckpt_path,
            elapsed_sec=0.0,
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = ppo_config or PPOConfig()
    preprocessor = ObservationPreprocessor(device)

    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=map_number,
        observation_mode=observation_mode,
        include_inventory=True,
        headless=True,
        max_steps=max_steps_per_episode,
    )
    obs_raw = env.reset()
    obs_dim = len(obs_raw)

    agent = PPOAgent(cfg).to(device)

    teacher = None
    # Load starting weights from BC checkpoint (or any prior checkpoint)
    if resume_from and resume_from.exists():
        ckpt = torch.load(str(resume_from), map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        
        # Initialize teacher for KL regularization to prevent catastrophic forgetting
        teacher = PPOAgent(cfg).to(device)
        teacher.load_state_dict(ckpt["agent_state_dict"])
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        
        if verbose:
            print(f"  [{level_tag}] Loaded warm-start from {resume_from.name} (KL coef: {teacher_kl_coef})")
    elif verbose:
        print(f"  [{level_tag}] Training from random init")

    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    return_rms = RunningMeanStd()

    rollout = RolloutBuffer(
        rollout_length=cfg.rollout_length,
        obs_dim=obs_dim,
        n_actions=agent.n_actions,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    obs_tensor = preprocessor(obs_raw)
    action_mask_np = env.get_valid_actions()
    action_mask_tensor = torch.tensor(action_mask_np, dtype=torch.bool, device=device)

    total_timesteps_done = 0
    episode_count = 0
    episode_reward = 0.0
    episode_successes: List[float] = []
    episode_collection_rates: List[float] = []
    best_success = 0.0
    best_path = ckpt_dir / f"{level_tag}_best.pt"
    start_time = time.time()
    done = False

    if verbose:
        print(
            f"  [{level_tag}] PPO training | {total_timesteps} steps | "
            f"device={device} | rollout={cfg.rollout_length}"
        )

    while total_timesteps_done < total_timesteps:
        # ── Collect rollout ──────────────────────────────────────────────
        rollout.reset()
        agent.eval()

        for _ in range(cfg.rollout_length):
            if total_timesteps_done >= total_timesteps:
                break

            # Epsilon-greedy: force a random valid action to escape BC determinism.
            # Stores log_prob under the current policy (not uniform), so the PPO
            # importance ratio correctly credits/penalizes the forced action.
            if explore_eps > 0 and np.random.random() < explore_eps:
                valid_acts = np.where(action_mask_np)[0]
                action = int(np.random.choice(valid_acts))
                with torch.no_grad():
                    feats_ep = agent.encoder(obs_tensor.unsqueeze(0))
                    dist_ep  = agent.policy(feats_ep, action_mask=action_mask_tensor.unsqueeze(0))
                    log_prob = float(dist_ep.log_prob(torch.tensor([action], device=device)).item())
                    value    = float(agent.value(feats_ep).item())
            else:
                action, log_prob, value = agent.select_action(obs_tensor, action_mask_tensor)
            next_obs_raw, reward, done, info = env.step(action)

            rollout.add(
                obs_raw.astype(np.int16), action, float(reward), done,
                log_prob, value, action_mask_np,
            )
            episode_reward += reward
            total_timesteps_done += 1

            if done:
                success = 1.0 if info.get("level_completed", False) else 0.0
                episode_successes.append(success)
                total_targets = info.get("total_targets")
                total_collected = info.get("total_collected")
                total_targets = total_targets if isinstance(total_targets, int) else 1
                total_collected = total_collected if isinstance(total_collected, int) else 0
                episode_collection_rates.append(total_collected / max(1, total_targets))
                episode_count += 1
                episode_reward = 0.0

                obs_raw = env.reset()
            else:
                obs_raw = next_obs_raw

            obs_tensor = preprocessor(obs_raw)
            action_mask_np = env.get_valid_actions()
            action_mask_tensor = torch.tensor(action_mask_np, dtype=torch.bool, device=device)

        # ── PPO update ───────────────────────────────────────────────────
        agent.train()

        # Compute GAE with return normalization
        with torch.no_grad():
            last_val = agent.value(agent.encoder(obs_tensor.unsqueeze(0))).item()
        rollout.compute_gae(last_val, last_done=done)

        advantages = rollout.advantages[:rollout.ptr]
        returns    = rollout.returns[:rollout.ptr]
        return_rms.update(returns)

        policy_losses, value_losses, entropies, kl_divs = [], [], [], []

        for _ in range(cfg.n_epochs):
            for batch in rollout.get_batches(cfg.minibatch_size):
                obs_b      = preprocessor.process_numpy_batch(batch["observations"]).to(device)
                actions_b  = torch.tensor(batch["actions"], dtype=torch.long, device=device)
                old_lp_b   = torch.tensor(batch["log_probs"], dtype=torch.float32, device=device)
                adv_b      = torch.tensor(batch["advantages"], dtype=torch.float32, device=device)
                # Normalize stored returns with running stats
                raw_ret    = batch["returns"]
                ret_b      = torch.tensor(
                    (raw_ret - return_rms.mean) / (return_rms.std + 1e-8),
                    dtype=torch.float32, device=device,
                )
                masks_b    = torch.tensor(batch["action_masks"], dtype=torch.bool, device=device)

                if cfg.normalize_advantages:
                    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                # Single encoder pass — features reused for policy, value, and KL.
                features_b = agent.encoder(obs_b)
                dist_b     = agent.policy(features_b, action_mask=masks_b)
                values_b   = agent.value(features_b)
                log_probs_b = dist_b.log_prob(actions_b)
                entropy_b   = dist_b.entropy()

                ratio = torch.exp(log_probs_b - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = cfg.value_coeff * nn.functional.mse_loss(values_b, ret_b)

                # KL divergence against teacher — reuses dist_b.logits, no second encoder pass.
                # KL anneals from kl_coef → 10% of kl_coef over the first 60% of training,
                # then holds at 10% so the BC anchor releases before exploration is needed.
                progress = min(1.0, total_timesteps_done / total_timesteps)
                kl_anneal = max(0.1, 1.0 - progress / 0.6)
                eff_kl_coef = teacher_kl_coef * kl_anneal

                kl_loss = torch.tensor(0.0, device=device)
                if teacher is not None and eff_kl_coef > 0:
                    with torch.no_grad():
                        t_features = teacher.encoder(obs_b)
                        t_dist = teacher.policy(t_features, action_mask=masks_b)
                        t_log_probs = torch.log_softmax(t_dist.logits, dim=-1)

                    s_log_probs = torch.log_softmax(dist_b.logits, dim=-1)
                    s_probs = s_log_probs.exp()
                    kl_per_action = s_probs * (s_log_probs - t_log_probs)
                    kl_per_action = torch.where(
                        torch.isfinite(kl_per_action),
                        kl_per_action,
                        torch.zeros_like(kl_per_action),
                    )
                    kl_loss = kl_per_action.sum(dim=-1).mean()

                # Entropy decays toward entropy_min; tripled when stuck at 0% past 15% of budget
                eff_entropy_coeff = cfg.entropy_coeff - progress * (cfg.entropy_coeff - cfg.entropy_min)
                if (progress > 0.15 and len(episode_successes) >= 5
                        and float(np.mean(episode_successes[-32:])) == 0.0):
                    eff_entropy_coeff = min(eff_entropy_coeff * 3.0, 0.4)

                entropy_loss = -eff_entropy_coeff * entropy_b.mean()

                loss = policy_loss + value_loss + entropy_loss + (eff_kl_coef * kl_loss)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_b.mean().item())
                if teacher is not None:
                    kl_divs.append(kl_loss.item())

        # ── Logging & best-model tracking ────────────────────────────────
        if total_timesteps_done % log_interval < cfg.rollout_length and episode_successes:
            window = 32
            recent_success = float(np.mean(episode_successes[-window:]))
            recent_coll = float(np.mean(episode_collection_rates[-window:])) if episode_collection_rates else 0.0
            if verbose:
                kl_str = f" | KL={np.mean(kl_divs):.4f}" if kl_divs else ""
                print(
                    f"    [{level_tag}] t={total_timesteps_done:>7d} | "
                    f"eps={episode_count:>5d} | "
                    f"success={recent_success:.1%} | "
                    f"coll={recent_coll:.1%} | "
                    f"π_loss={np.mean(policy_losses):.4f} | "
                    f"v_loss={np.mean(value_losses):.4f} | "
                    f"H={np.mean(entropies):.3f}{kl_str}"
                )
            if recent_success > best_success:
                best_success = recent_success
                torch.save(
                    {
                        "agent_state_dict": agent.state_dict(),
                        "level_kind": map_kind,
                        "level_num": map_number,
                        "phase": "rl",
                        "total_timesteps": total_timesteps_done,
                        "episode_count": episode_count,
                        "best_success": best_success,
                        "ppo_config": cfg,
                    },
                    str(best_path),
                )

    # ── Final checkpoint ──────────────────────────────────────────────────
    final_success = float(np.mean(episode_successes[-32:])) if episode_successes else 0.0
    elapsed = time.time() - start_time

    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "level_kind": map_kind,
            "level_num": map_number,
            "phase": "rl",
            "total_timesteps": total_timesteps_done,
            "episode_count": episode_count,
            "best_success": best_success,
            "final_success": final_success,
            "elapsed_sec": elapsed,
            "ppo_config": cfg,
        },
        str(final_ckpt_path),
    )
    if verbose:
        print(
            f"  [{level_tag}] Done | best={best_success:.1%} | final={final_success:.1%} | "
            f"t={elapsed:.0f}s → {final_ckpt_path}"
        )

    env.close()
    return LevelTrainResult(
        map_kind=map_kind,
        map_number=map_number,
        timesteps=total_timesteps_done,
        episodes=episode_count,
        best_success=best_success,
        final_success=final_success,
        checkpoint_path=final_ckpt_path,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Batch per-level training
# ---------------------------------------------------------------------------

def train_all_levels(
    levels: List[Tuple[str, int]],
    ckpt_dir: Path,
    bc_dir: Optional[Path] = None,
    total_timesteps_per_level: int = 200_000,
    sequential: bool = False,
    ppo_config: Optional[PPOConfig] = None,
    max_steps_per_episode: int = 1500,
    observation_mode: str = "full",
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    verbose: bool = True,
    teacher_kl_coef: float = 0.05,
    explore_eps: float = 0.10,
) -> List[LevelTrainResult]:
    """Train PPO on each level sequentially, saving a checkpoint for each.

    Parameters
    ----------
    levels        : list of (kind, number) pairs to train
    ckpt_dir      : directory where *_final.pt checkpoints are written
    bc_dir        : directory containing *_bc.pt behavioral-cloning checkpoints
                    (used as warm-start for every level when sequential=False)
    sequential    : if True, each level starts from the PREVIOUS level's final
                    checkpoint instead of the BC checkpoint
    total_timesteps_per_level : PPO steps to run per level
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results: List[LevelTrainResult] = []
    prev_ckpt: Optional[Path] = None

    print(
        f"\n[PerLevelTrain] Training {len(levels)} levels | "
        f"{total_timesteps_per_level} steps/level | sequential={sequential}"
    )

    for kind, num in levels:
        level_tag = f"{kind}_{num:02d}"

        # Determine warm-start checkpoint
        if sequential and prev_ckpt is not None:
            warm_start = prev_ckpt
        elif bc_dir is not None:
            bc_ckpt = bc_dir / f"{level_tag}_bc.pt"
            if not bc_ckpt.exists():
                # Fall back to joint BC checkpoint
                joint_bc = bc_dir / "joint_bc.pt"
                bc_ckpt = joint_bc if joint_bc.exists() else None
            warm_start = bc_ckpt
        else:
            warm_start = None

        result = train_ppo_single_level(
            map_kind=kind,
            map_number=num,
            ckpt_dir=ckpt_dir,
            total_timesteps=total_timesteps_per_level,
            resume_from=warm_start,
            ppo_config=ppo_config,
            max_steps_per_episode=max_steps_per_episode,
            observation_mode=observation_mode,
            device=device,
            seed=seed + num,
            verbose=verbose,
            teacher_kl_coef=teacher_kl_coef,
            explore_eps=explore_eps,
        )
        results.append(result)

        if sequential:
            prev_ckpt = result.checkpoint_path

    # Summary
    print(f"\n[PerLevelTrain] All levels complete:")
    print(f"  {'Level':<12} {'Best':<8} {'Final':<8} {'Steps':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
    for r in results:
        print(
            f"  {r.map_kind}_{r.map_number:02d}    "
            f"{r.best_success:.1%}   {r.final_success:.1%}   {r.timesteps:>10,}"
        )

    # Write summary CSV
    summary_path = ckpt_dir / "per_level_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level_kind", "level_num", "timesteps", "episodes",
                         "best_success", "final_success", "elapsed_sec", "checkpoint"])
        for r in results:
            writer.writerow([
                r.map_kind, r.map_number, r.timesteps, r.episodes,
                f"{r.best_success:.4f}", f"{r.final_success:.4f}",
                f"{r.elapsed_sec:.1f}", str(r.checkpoint_path),
            ])
    print(f"\n  Summary written to {summary_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_level_range(s: str) -> List[Tuple[str, int]]:
    levels = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            levels.extend(("normal", i) for i in range(int(lo), int(hi) + 1))
        else:
            levels.append(("normal", int(part)))
    return levels


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Per-level PPO training with individual level checkpoints."
    )
    parser.add_argument("--levels", default="1-30",
                        help="Levels to train (e.g. '1-30'). Default: 1-30.")
    parser.add_argument("--ckpt-dir", default="checkpoints/per_level",
                        help="Output directory for *_final.pt checkpoints.")
    parser.add_argument("--bc-dir", default=None,
                        help="BC checkpoint directory for warm-starts.")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="PPO steps per level. Default: 200000.")
    parser.add_argument("--sequential", action="store_true",
                        help="Chain levels: each starts from the previous level's final checkpoint.")
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="Max env steps per episode. Default: 1500.")
    parser.add_argument("--obs-mode", default="full",
                        help="Observation mode: 'full', 'compact', or 'local'. Default: full.")
    parser.add_argument("--device", default="auto",
                        help="'cpu', 'cuda', or 'auto'. Default: auto.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (default 3e-5 for fine-tuning)")
    parser.add_argument("--entropy", type=float, default=0.05, help="Entropy coeff (default 0.05)")
    parser.add_argument("--entropy-min", type=float, default=0.01, help="Entropy coeff floor (default 0.01)")
    parser.add_argument("--kl-coef", type=float, default=0.3, help="KL divergence coeff for teacher policy (default 0.3)")
    parser.add_argument("--rollout", type=int, default=2048,
                        help="Rollout buffer size. Default: 2048.")
    parser.add_argument("--explore-eps", type=float, default=0.10,
                        help="Epsilon for forced random actions during rollout (default 0.10).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-epoch output.")
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[PerLevelTrain] Using device: {device}")

    ppo_cfg = PPOConfig(
        lr=args.lr,
        entropy_coeff=args.entropy,
        entropy_min=args.entropy_min,
        rollout_length=args.rollout,
    )

    levels = _parse_level_range(args.levels)
    train_all_levels(
        levels=levels,
        ckpt_dir=Path(args.ckpt_dir),
        bc_dir=Path(args.bc_dir) if args.bc_dir else None,
        total_timesteps_per_level=args.timesteps,
        sequential=args.sequential,
        ppo_config=ppo_cfg,
        max_steps_per_episode=args.max_steps,
        observation_mode=args.obs_mode,
        device=device,
        seed=args.seed,
        verbose=not args.quiet,
        teacher_kl_coef=args.kl_coef,
        explore_eps=args.explore_eps,
    )


if __name__ == "__main__":
    main()
