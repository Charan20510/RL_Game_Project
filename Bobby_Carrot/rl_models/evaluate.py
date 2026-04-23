"""Multi-level evaluation harness for trained RL agents.

Loads a saved checkpoint, runs deterministic episodes on all test levels,
and reports per-level metrics.

Usage:
    python -m Bobby_Carrot.rl_models.evaluate --algo ppo --checkpoint checkpoints/ppo/ppo_final.pt
    python -m Bobby_Carrot.rl_models.evaluate --algo rainbow --checkpoint checkpoints/rainbow/rainbow_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from collections import deque

import numpy as np
import torch

# Ensure imports work
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_GAME_PYTHON = _PROJECT_ROOT / "Game_Python"
if str(_GAME_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GAME_PYTHON))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Bobby_Carrot.rl_models.config import PPOConfig, RainbowConfig, LevelConfig
from Bobby_Carrot.rl_models.networks import ObservationPreprocessor
from Bobby_Carrot.rl_models.ppo import PPOAgent
from Bobby_Carrot.rl_models.rainbow import RainbowAgent


def _bfs_distance(data: list, start: Tuple[int, int], goals: set) -> int:
    """Shortest walkable-tile distance from start to any goal. inf if unreachable.

    Walkable: tile >= 18 AND tile not in {31, 46} (hole/collected-egg trap).
    Crumble tiles (30) are treated as walkable here — we want the ideal path
    assuming correct crumble traversal order.
    """
    if start in goals:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (cx, cy), dist = queue.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < 16 and 0 <= ny < 16 and (nx, ny) not in visited:
                tile = data[nx + ny * 16]
                if tile >= 18 and tile != 31 and tile != 46:
                    if (nx, ny) in goals:
                        return dist + 1
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
    return 10**9  # Effectively unreachable — caller should treat as inf


def compute_optimal_path_length(env) -> int:
    """Greedy nearest-target TSP approximation: start → nearest carrot → ... → finish.

    Used as the denominator for `steps_to_solve / shortest_path` ratio in eval.
    Exact TSP is NP-hard, but greedy nearest-neighbour on a 16×16 grid with
    few (<20) targets is close enough to give a meaningful overhead metric.
    Returns 10**9 when no feasible plan exists (unreachable targets / finish).
    """
    data = list(env.map_info.data)
    pos = env.bobby.coord_src
    targets = set()
    finish = set()
    for y in range(16):
        for x in range(16):
            t = data[x + y * 16]
            if t == 19 or t == 45:
                targets.add((x, y))
            elif t == 44:
                finish.add((x, y))

    total = 0
    while targets:
        best_dist = 10**9
        best_tgt = None
        for tgt in targets:
            d = _bfs_distance(data, pos, {tgt})
            if d < best_dist:
                best_dist = d
                best_tgt = tgt
        if best_tgt is None or best_dist >= 10**9:
            return 10**9
        total += best_dist
        pos = best_tgt
        targets.remove(best_tgt)
        # Simulate collection: tile 19 → 20, tile 45 → 46 (hazard).
        idx = pos[0] + pos[1] * 16
        if data[idx] == 19:
            data[idx] = 20
        elif data[idx] == 45:
            data[idx] = 46

    if not finish:
        return total
    final_leg = _bfs_distance(data, pos, finish)
    if final_leg >= 10**9:
        return 10**9
    return total + final_leg


_LEVEL_MECHANIC_LABELS: Dict[int, str] = {
    1:  "floor+carrot",
    2:  "crumble-intro",
    3:  "crumble",
    4:  "crumble-dense",
    5:  "crumble",
    6:  "crumble",
    7:  "crumble-dense",
    8:  "conveyor-LR",
    9:  "conveyor-RD",
    10: "conveyor-UD",
    11: "conveyor-LU",
    12: "conveyor-all",
    13: "bidir-conv",
    14: "arrow+bidir",
    15: "arrow+conv",
    16: "switch+conv",
    17: "switch+conv",
    18: "arrow+key+lock",
    19: "arrow+switch",
    20: "key+lock",
    21: "switch+conv",
    22: "switch+key+conv",
    23: "switch+arrow",
    24: "arrow+conv",
    25: "switch+key+conv",
    26: "key+lock+65c",   # 65 carrots — game maximum
    27: "arrow+switch",
    28: "conv+switch",
    29: "arrow+switch",
    30: "switch+key+conv",
}


def evaluate_agent(
    algo: str,
    checkpoint_path: str,
    levels: List[Tuple[str, int]],
    episodes_per_level: int = 10,
    max_steps: int = 300,
    observation_mode: str = "full",
    device_str: str = "auto",
    render: bool = False,
    render_fps: float = 4.0,
    save_frames: bool = False,
    frames_dir: str = "frames",
    use_mcts: bool = False,
    mcts_sims: int = 128,
    mcts_depth: int = 25,
    forgetting_levels: Optional[List[Tuple[str, int]]] = None,
) -> Dict[str, Any]:
    """Evaluate a trained agent across multiple levels.

    Returns a dict with per-level and aggregate metrics.
    """
    from bobby_carrot.rl_env import BobbyCarrotEnv  # type: ignore

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create agent with matching config from checkpoint if available
    if algo == "ppo":
        saved_config = None
        if "config" in ckpt and "ppo" in ckpt["config"]:
            saved_config = ckpt["config"]["ppo"]
        agent = PPOAgent(saved_config or PPOConfig()).to(device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        agent.eval()
    elif algo == "rainbow":
        agent = RainbowAgent(RainbowConfig()).to(device)
        agent.load_state_dict(ckpt["online_state_dict"])
        agent.eval()
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    preprocessor = ObservationPreprocessor(device)

    # Optionally build MCTS evaluator (PPO only)
    mcts_evaluator = None
    if use_mcts:
        if algo != "ppo":
            print("Warning: MCTS is only supported for PPO — ignoring --use-mcts for Rainbow.")
        else:
            from Bobby_Carrot.rl_models.mcts_eval import MCTSEvaluator
            mcts_evaluator = MCTSEvaluator(
                agent, preprocessor,  # type: ignore[arg-type]
                n_sims=mcts_sims,
                max_depth=mcts_depth,
                device=device,
            )
            print(f"  MCTS enabled: {mcts_sims} sims × depth {mcts_depth}")

    # Setup frames directory if saving frames
    frames_path = Path(frames_dir)
    if save_frames:
        frames_path.mkdir(parents=True, exist_ok=True)

    # Per-level results
    results: Dict[str, Dict[str, Any]] = {}
    all_successes: List[float] = []
    all_rewards: List[float] = []
    all_steps: List[float] = []
    all_collected: List[float] = []

    mode_tag = "MCTS" if mcts_evaluator else algo.upper()
    print(f"\n{'='*72}")
    print(f"  Evaluating {mode_tag} on {len(levels)} levels × {episodes_per_level} episodes")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"{'='*72}\n")

    print(f"{'Level':<15} {'Mechanic':<18} {'Success%':>9} {'Collected%':>11} {'Avg Reward':>11} {'Avg Steps':>10} {'PathRatio':>10}")
    print("-" * 87)

    all_path_ratios: List[float] = []

    for kind, num in levels:
        level_key = f"{kind}-{num:02d}"
        mechanic_label = _LEVEL_MECHANIC_LABELS.get(num, "—") if kind == "normal" else "egg-level"
        successes: List[float] = []
        rewards: List[float] = []
        steps_list: List[float] = []
        collected: List[float] = []
        path_ratios: List[float] = []

        env = BobbyCarrotEnv(
            map_kind=kind,
            map_number=num,
            observation_mode=observation_mode,
            include_inventory=True,
            headless=not render,
            max_steps=max_steps,
        )

        for ep in range(episodes_per_level):
            obs_raw = env.reset()
            optimal_len = compute_optimal_path_length(env)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            info: Dict[str, Any] = {}

            if render:
                env.render()
                if save_frames:
                    import pygame
                    frame_path = frames_path / f"{level_key}_ep{ep+1}_step{ep_steps:04d}.png"
                    screen = getattr(env, '_screen', None)
                    if screen is not None:
                        pygame.image.save(screen, str(frame_path))  # type: ignore[arg-type]

            while not done:
                if mcts_evaluator is not None:
                    action = mcts_evaluator.select_action(env, obs_raw)
                else:
                    obs_t = preprocessor(obs_raw)
                    mask_np = env.get_valid_actions()
                    mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device)
                    with torch.no_grad():
                        if algo == "ppo":
                            features = agent.encoder(obs_t.unsqueeze(0))  # type: ignore[union-attr]
                            dist = agent.policy(features, action_mask=mask_t.unsqueeze(0))  # type: ignore[union-attr]
                            action = int(dist.probs.argmax(dim=-1).item())
                        else:  # rainbow
                            q = agent.q_values(obs_t.unsqueeze(0))  # type: ignore[union-attr]
                            action = int(q.argmax(dim=-1).item())

                obs_raw, reward, done, info = env.step(action)
                ep_reward += reward
                ep_steps += 1

                if render:
                    env.render()
                    if save_frames:
                        import pygame
                        frame_path = frames_path / f"{level_key}_ep{ep+1}_step{ep_steps:04d}.png"
                        screen = getattr(env, '_screen', None)
                        if screen is not None:
                            pygame.image.save(screen, str(frame_path))  # type: ignore[arg-type]
                    import time
                    time.sleep(1.0 / render_fps)

            successful = bool(info.get("level_completed", False))
            successes.append(1.0 if successful else 0.0)
            collected.append(1.0 if info.get("all_collected", False) else 0.0)
            rewards.append(ep_reward)
            steps_list.append(float(ep_steps))
            if successful and optimal_len < 10**9 and optimal_len > 0:
                path_ratios.append(ep_steps / float(optimal_len))

        env.close()

        level_success = float(np.mean(successes))
        level_collected = float(np.mean(collected))
        level_reward = float(np.mean(rewards))
        level_steps = float(np.mean(steps_list))
        level_ratio = float(np.mean(path_ratios)) if path_ratios else float("nan")

        results[level_key] = {
            "success_rate": level_success,
            "collection_rate": level_collected,
            "avg_reward": level_reward,
            "avg_steps": level_steps,
            "path_ratio": level_ratio,
            "mechanic": mechanic_label,
        }

        all_successes.extend(successes)
        all_rewards.extend(rewards)
        all_steps.extend(steps_list)
        all_collected.extend(collected)
        all_path_ratios.extend(path_ratios)

        ratio_str = f"{level_ratio:>9.2f}" if not np.isnan(level_ratio) else f"{'n/a':>9}"
        print(
            f"{level_key:<15} {mechanic_label:<18} {level_success:>8.1%} "
            f"{level_collected:>10.1%} {level_reward:>10.2f} {level_steps:>9.1f} {ratio_str}"
        )

    # Aggregate
    print("-" * 87)
    agg_success = float(np.mean(all_successes))
    agg_collected = float(np.mean(all_collected))
    agg_reward = float(np.mean(all_rewards))
    agg_steps = float(np.mean(all_steps))
    agg_ratio = float(np.mean(all_path_ratios)) if all_path_ratios else float("nan")
    agg_ratio_str = f"{agg_ratio:>9.2f}" if not np.isnan(agg_ratio) else f"{'n/a':>9}"
    print(
        f"{'AGGREGATE':<15} {'':<18} {agg_success:>8.1%} "
        f"{agg_collected:>10.1%} {agg_reward:>10.2f} {agg_steps:>9.1f} {agg_ratio_str}"
    )
    print()

    output: Dict[str, Any] = {
        "per_level": results,
        "aggregate": {
            "success_rate": agg_success,
            "collection_rate": agg_collected,
            "avg_reward": agg_reward,
            "avg_steps": agg_steps,
            "path_ratio": agg_ratio,
            "total_episodes": len(all_successes),
        },
    }

    # Forgetting metric: re-evaluate on early training levels to check retention
    if forgetting_levels:
        print(f"\n{'='*72}")
        print("  FORGETTING CHECK — early training levels")
        print(f"{'='*72}")
        forgetting_results = evaluate_agent(
            algo=algo,
            checkpoint_path=checkpoint_path,
            levels=forgetting_levels,
            episodes_per_level=episodes_per_level,
            max_steps=max_steps,
            observation_mode=observation_mode,
            device_str=device_str,
            render=False,
            use_mcts=False,
        )
        output["forgetting"] = forgetting_results

    return output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Bobby Carrot RL agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo", type=str, required=True, choices=["ppo", "rainbow"],
                   help="Algorithm to evaluate")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint file (.pt)")
    p.add_argument("--episodes", type=int, default=10,
                   help="Episodes per level")
    p.add_argument("--max-steps", type=int, default=300,
                   help="Max steps per episode")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto, cuda, cpu")
    p.add_argument("--render", action="store_true",
                   help="Render episodes visually")
    p.add_argument("--render-fps", type=float, default=4.0,
                   help="Frames per second for rendering")
    p.add_argument("--save-frames", action="store_true",
                   help="Save rendered frames to disk as PNG images")
    p.add_argument("--frames-dir", type=str, default="frames",
                   help="Directory to save the rendered frames")

    # Level selection
    p.add_argument("--eval-set", type=str, default="test", choices=["test", "train", "all"],
                   help="Which level set to evaluate on")
    p.add_argument("--levels", type=str, default=None,
                   help="Comma-separated list of levels (e.g. 'normal-1,egg-5')")

    # MCTS
    p.add_argument("--use-mcts", action="store_true", default=False,
                   help="Use MCTS at test time (PPO only)")
    p.add_argument("--mcts-sims", type=int, default=128,
                   help="Number of MCTS simulations per step")
    p.add_argument("--mcts-depth", type=int, default=25,
                   help="Maximum MCTS rollout depth")

    # Forgetting metric
    p.add_argument("--check-forgetting", action="store_true", default=False,
                   help="Also evaluate on early train levels (1-5) to check retention")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    level_config = LevelConfig()

    # Determine levels
    if args.levels:
        levels = []
        for spec in args.levels.split(","):
            spec = spec.strip()
            if "-" in spec:
                kind, num = spec.rsplit("-", 1)
                levels.append((kind, int(num)))
            else:
                levels.append(("normal", int(spec)))
    elif args.eval_set == "test":
        levels = level_config.test_levels
    elif args.eval_set == "train":
        levels = level_config.train_levels
    else:
        levels = level_config.train_levels + level_config.test_levels

    forgetting_levels = (
        [("normal", i) for i in range(1, 6)]
        if args.check_forgetting else None
    )

    evaluate_agent(
        algo=args.algo,
        checkpoint_path=args.checkpoint,
        levels=levels,
        episodes_per_level=args.episodes,
        max_steps=args.max_steps,
        device_str=args.device,
        render=args.render,
        render_fps=args.render_fps,
        save_frames=args.save_frames,
        use_mcts=args.use_mcts,
        mcts_sims=args.mcts_sims,
        mcts_depth=args.mcts_depth,
        forgetting_levels=forgetting_levels,
        frames_dir=args.frames_dir,
    )


if __name__ == "__main__":
    main()
