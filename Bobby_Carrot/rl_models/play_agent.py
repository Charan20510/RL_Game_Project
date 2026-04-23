"""Deployment & visualization: load a checkpoint and watch the agent play.

Loads any checkpoint produced by bc_pretrain.py or per_level_train.py and
runs it in the Bobby Carrot pygame GUI so you can visually inspect behavior.

Usage:
    # Watch the agent on the level it was trained for (auto-detected)
    python -m Bobby_Carrot.rl_models.play_agent \\
        --checkpoint checkpoints/per_level/normal_05_final.pt

    # Override the level
    python -m Bobby_Carrot.rl_models.play_agent \\
        --checkpoint checkpoints/bc/joint_bc.pt --kind normal --level 12

    # Run multiple episodes
    python -m Bobby_Carrot.rl_models.play_agent \\
        --checkpoint checkpoints/per_level/normal_07_final.pt --episodes 5

    # Slower rendering (easier to follow)
    python -m Bobby_Carrot.rl_models.play_agent \\
        --checkpoint checkpoints/per_level/normal_03_final.pt --fps 3

    # Headless benchmark (no GUI, just metrics)
    python -m Bobby_Carrot.rl_models.play_agent \\
        --checkpoint checkpoints/per_level/normal_01_final.pt --no-render --episodes 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

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
from Bobby_Carrot.rl_models.config import PPOConfig
from Bobby_Carrot.rl_models.networks import ObservationPreprocessor
from Bobby_Carrot.rl_models.ppo import PPOAgent


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_agent(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
) -> Tuple[PPOAgent, ObservationPreprocessor, Dict]:
    """Load a PPOAgent from a checkpoint file.

    Returns (agent, preprocessor, metadata_dict).
    The metadata dict contains keys like 'level_kind', 'level_num', 'phase',
    'best_success', 'final_success', etc., depending on how the checkpoint
    was saved.
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    # Reconstruct PPOConfig — stored in checkpoint or use defaults
    cfg = ckpt.get("ppo_config", PPOConfig())
    if not isinstance(cfg, PPOConfig):
        cfg = PPOConfig()

    agent = PPOAgent(cfg).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    preprocessor = ObservationPreprocessor(device)

    metadata = {
        "level_kind":    ckpt.get("level_kind", "normal"),
        "level_num":     ckpt.get("level_num", 1),
        "phase":         ckpt.get("phase", "unknown"),
        "best_success":  ckpt.get("best_success", 0.0),
        "final_success": ckpt.get("final_success", 0.0),
        "total_timesteps": ckpt.get("total_timesteps", 0),
    }
    return agent, preprocessor, metadata


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(
    agent: PPOAgent,
    preprocessor: ObservationPreprocessor,
    env: BobbyCarrotEnv,
    device: torch.device,
    render: bool = True,
    fps: float = 5.0,
    deterministic: bool = True,
) -> Dict:
    """Run one episode of the agent in *env*.

    Returns a metrics dict with keys: solved, steps, collected, total_targets.
    """
    obs_raw = env.reset()
    done = False
    step_count = 0
    frame_delay = 1.0 / fps if fps > 0 else 0.0

    if render:
        env.render()
        time.sleep(0.5)   # brief pause so the initial frame is visible

    while not done:
        obs_tensor = preprocessor(obs_raw).to(device)
        action_mask = torch.tensor(
            env.get_valid_actions(), dtype=torch.bool, device=device
        )

        with torch.no_grad():
            features = agent.encoder(obs_tensor.unsqueeze(0))
            dist = agent.policy(features, action_mask=action_mask.unsqueeze(0))
            if deterministic:
                action = int(dist.probs.argmax(dim=-1).item())
            else:
                action = int(dist.sample().item())

        obs_raw, _reward, done, info = env.step(action)
        step_count += 1

        if render:
            env.render()
            if frame_delay > 0:
                time.sleep(frame_delay)

            # Pump pygame events so the window stays responsive
            try:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        done = True
                        break
            except ImportError:
                pass

    solved = bool(info.get("level_completed", False))
    total_collected = int(info.get("total_collected", 0))
    total_targets   = int(info.get("total_targets", 0))

    return {
        "solved":          solved,
        "steps":           step_count,
        "collected":       total_collected,
        "total_targets":   total_targets,
        "collection_rate": total_collected / max(1, total_targets),
    }


# ---------------------------------------------------------------------------
# Multi-episode play
# ---------------------------------------------------------------------------

def play(
    checkpoint_path: Path,
    map_kind: Optional[str] = None,
    map_number: Optional[int] = None,
    n_episodes: int = 1,
    fps: float = 5.0,
    max_steps: int = 1500,
    render: bool = True,
    deterministic: bool = True,
    device_str: str = "auto",
    observation_mode: str = "full",
    verbose: bool = True,
) -> List[Dict]:
    """Load a checkpoint and run the agent for *n_episodes*.

    Level is auto-detected from the checkpoint unless overridden.
    Returns a list of per-episode metric dicts.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    agent, preprocessor, meta = load_agent(checkpoint_path, device)

    kind   = map_kind   if map_kind   is not None else str(meta["level_kind"])
    number = map_number if map_number is not None else int(meta["level_num"])

    # Sanity check: fallback to level 1 if checkpoint has no level info
    if number <= 0:
        number = 1

    if verbose:
        print(f"\n[PlayAgent] Checkpoint: {checkpoint_path.name}")
        print(f"  Phase:         {meta['phase']}")
        print(f"  Level:         {kind} {number}")
        print(f"  Best success:  {meta['best_success']:.1%}")
        print(f"  Total steps trained: {meta['total_timesteps']:,}")
        print(f"  Rendering: {render} | FPS: {fps} | Episodes: {n_episodes}")

    env = BobbyCarrotEnv(
        map_kind=kind,
        map_number=number,
        observation_mode=observation_mode,
        include_inventory=True,
        headless=not render,
        max_steps=max_steps,
    )

    results: List[Dict] = []
    solved_count = 0

    for ep in range(1, n_episodes + 1):
        metrics = run_episode(
            agent, preprocessor, env, device,
            render=render, fps=fps, deterministic=deterministic,
        )
        results.append(metrics)
        if metrics["solved"]:
            solved_count += 1

        if verbose:
            print(
                f"  Episode {ep:>3d}: {'SOLVED' if metrics['solved'] else 'failed'} | "
                f"steps={metrics['steps']:>5d} | "
                f"collected={metrics['collected']}/{metrics['total_targets']}"
            )

        if render and ep < n_episodes:
            time.sleep(1.0)   # brief pause between episodes

    if verbose and n_episodes > 1:
        success_rate = solved_count / n_episodes
        avg_steps = float(np.mean([r["steps"] for r in results]))
        avg_coll  = float(np.mean([r["collection_rate"] for r in results]))
        print(
            f"\n  Summary: {solved_count}/{n_episodes} solved ({success_rate:.1%}) | "
            f"avg_steps={avg_steps:.0f} | avg_collection={avg_coll:.1%}"
        )

    env.close()
    return results


# ---------------------------------------------------------------------------
# Level-list sweep (benchmark all per-level checkpoints)
# ---------------------------------------------------------------------------

def benchmark_checkpoints(
    ckpt_dir: Path,
    levels: Optional[List[Tuple[str, int]]] = None,
    n_episodes: int = 10,
    max_steps: int = 1500,
    device_str: str = "auto",
    observation_mode: str = "full",
) -> None:
    """Run the saved *_final.pt checkpoint for each level and print a summary table."""
    if levels is None:
        levels = [("normal", i) for i in range(1, 31)]

    print(f"\n[Benchmark] {ckpt_dir}")
    print(f"  {'Level':<12} {'SuccessRate':>12} {'AvgSteps':>10} {'AvgColl':>10} {'Checkpoint'}")
    print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*30}")

    for kind, num in levels:
        level_tag = f"{kind}_{num:02d}"
        ckpt_path = ckpt_dir / f"{level_tag}_final.pt"
        if not ckpt_path.exists():
            print(f"  {level_tag:<12} {'(no ckpt)':>12}")
            continue

        results = play(
            checkpoint_path=ckpt_path,
            n_episodes=n_episodes,
            render=False,
            max_steps=max_steps,
            device_str=device_str,
            observation_mode=observation_mode,
            verbose=False,
        )
        success_rate = float(np.mean([r["solved"] for r in results]))
        avg_steps    = float(np.mean([r["steps"]   for r in results]))
        avg_coll     = float(np.mean([r["collection_rate"] for r in results]))
        print(
            f"  {level_tag:<12} {success_rate:>11.1%} {avg_steps:>10.0f} "
            f"{avg_coll:>9.1%}  {ckpt_path.name}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Load a Bobby Carrot RL checkpoint and visualize the agent."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint file (BC or per-level RL).",
    )
    parser.add_argument(
        "--kind", default=None,
        help="Override map kind ('normal'). Auto-detected from checkpoint if omitted.",
    )
    parser.add_argument(
        "--level", type=int, default=None,
        help="Override level number. Auto-detected from checkpoint if omitted.",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to run. Default: 1.",
    )
    parser.add_argument(
        "--fps", type=float, default=5.0,
        help="Rendering speed in frames per second. Default: 5.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1500,
        help="Max steps per episode. Default: 1500.",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Run headless (no GUI) for benchmarking.",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Sample actions stochastically instead of greedy argmax.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="'cpu', 'cuda', or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Benchmark all *_final.pt checkpoints in the checkpoint directory.",
    )
    parser.add_argument(
        "--obs-mode", default="full",
        help="Observation mode. Default: full.",
    )
    args = parser.parse_args(argv)

    ckpt_path = Path(args.checkpoint)

    if args.benchmark:
        benchmark_checkpoints(
            ckpt_dir=ckpt_path if ckpt_path.is_dir() else ckpt_path.parent,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            device_str=args.device,
            observation_mode=args.obs_mode,
        )
    else:
        play(
            checkpoint_path=ckpt_path,
            map_kind=args.kind,
            map_number=args.level,
            n_episodes=args.episodes,
            fps=args.fps,
            max_steps=args.max_steps,
            render=not args.no_render,
            deterministic=not args.stochastic,
            device_str=args.device,
            observation_mode=args.obs_mode,
            verbose=True,
        )


if __name__ == "__main__":
    main()
