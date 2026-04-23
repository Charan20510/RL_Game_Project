"""Expert dataset generation for all normal levels.

Runs ExpertSolver on each level, captures (obs, action) pairs, and saves
compressed numpy archives — one file per level — ready for BC pre-training.

Usage:
    # Generate datasets for all 30 normal levels
    python -m Bobby_Carrot.rl_models.dataset_gen

    # Only levels 1-10
    python -m Bobby_Carrot.rl_models.dataset_gen --levels 1-10

    # Custom output directory
    python -m Bobby_Carrot.rl_models.dataset_gen --output-dir my_expert_data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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
from Bobby_Carrot.rl_models.expert_solver import ExpertSolver


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def save_dataset(
    trajectory: List[Tuple[np.ndarray, int]],
    path: Path,
) -> None:
    """Save a trajectory as a compressed .npz file."""
    if not trajectory:
        raise ValueError("Cannot save empty trajectory.")
    observations = np.array([obs for obs, _ in trajectory], dtype=np.int16)
    actions      = np.array([act for _, act in trajectory], dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), observations=observations, actions=actions)


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a saved dataset. Returns (observations, actions).

    observations : int16 array of shape (N, obs_dim)
    actions      : int8  array of shape (N,)
    """
    data = np.load(str(path))
    return data["observations"], data["actions"]


# ---------------------------------------------------------------------------
# Single-level generation
# ---------------------------------------------------------------------------

def generate_level_dataset(
    map_kind: str,
    map_number: int,
    output_dir: Path,
    max_attempts: int = 5,
    max_steps: int = 2000,
    stuck_threshold: int = 40,
    min_steps: int = 10,
    verbose: bool = True,
) -> Dict:
    """Run ExpertSolver on one level and save the resulting dataset.

    Returns a dict with keys: solved, steps, attempts, path.
    """
    level_tag = f"{map_kind}_{map_number:02d}"
    out_path = output_dir / f"{level_tag}.npz"

    # Skip levels already generated
    if out_path.exists():
        obs, acts = load_dataset(out_path)
        if verbose:
            print(f"  [{level_tag}] already exists ({len(acts)} steps), skipping.")
        return {"solved": True, "steps": len(acts), "attempts": 0, "path": out_path}

    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=map_number,
        observation_mode="full",
        include_inventory=True,
        headless=True,
        max_steps=max_steps + 200,   # give solver a little more room than solver's own limit
    )

    best_trajectory: List[Tuple[np.ndarray, int]] = []
    solved = False

    for attempt in range(1, max_attempts + 1):
        solver = ExpertSolver(env, max_steps=max_steps, stuck_threshold=stuck_threshold)
        trajectory, attempt_solved = solver.solve()

        if attempt_solved:
            best_trajectory = trajectory
            solved = True
            break

        # Keep the attempt with the most items collected as fallback
        if len(trajectory) > len(best_trajectory):
            best_trajectory = trajectory

    if verbose:
        status = "SOLVED" if solved else "PARTIAL"
        print(
            f"  [{level_tag}] {status} | steps={len(best_trajectory)} "
            f"| attempts={min(attempt, max_attempts)}"
        )

    if best_trajectory and len(best_trajectory) >= min_steps:
        save_dataset(best_trajectory, out_path)
    elif best_trajectory:
        if verbose:
            print(f"  [{level_tag}] Trajectory too short ({len(best_trajectory)} steps < {min_steps}), discarding.")
        best_trajectory = []

    return {
        "solved": solved,
        "steps": len(best_trajectory),
        "attempts": min(attempt, max_attempts),
        "path": out_path if best_trajectory else None,
    }


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_all_datasets(
    levels: List[Tuple[str, int]],
    output_dir: Path,
    max_attempts: int = 5,
    max_steps: int = 2000,
    stuck_threshold: int = 40,
    verbose: bool = True,
) -> Dict[Tuple[str, int], Dict]:
    """Generate expert datasets for a list of (kind, number) levels.

    Returns a summary dict keyed by (kind, number).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[Tuple[str, int], Dict] = {}

    print(f"\n[DatasetGen] Generating expert data for {len(levels)} levels → {output_dir}")
    t0 = time.time()

    solved_count = 0
    for kind, num in levels:
        result = generate_level_dataset(
            map_kind=kind,
            map_number=num,
            output_dir=output_dir,
            max_attempts=max_attempts,
            max_steps=max_steps,
            stuck_threshold=stuck_threshold,
            verbose=verbose,
        )
        summary[(kind, num)] = result
        if result["solved"]:
            solved_count += 1

    elapsed = time.time() - t0
    print(
        f"\n[DatasetGen] Done in {elapsed:.1f}s — "
        f"{solved_count}/{len(levels)} levels solved completely."
    )
    unsolved = [(k, n) for (k, n), r in summary.items() if not r["solved"]]
    if unsolved:
        print(f"  Unsolved levels (partial data saved): {unsolved}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_level_range(s: str) -> List[Tuple[str, int]]:
    """Parse a range string like '1-30' or '1,3,5' into a list of (kind, num)."""
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
        description="Generate expert demonstration datasets for Bobby Carrot levels."
    )
    parser.add_argument(
        "--levels", default="1-30",
        help="Levels to generate (e.g. '1-30', '1-10', '1,3,5'). Default: 1-30.",
    )
    parser.add_argument(
        "--output-dir", default="expert_data",
        help="Directory to save .npz dataset files. Default: expert_data/",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=5,
        help="Solver restart attempts per level. Default: 5.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=2000,
        help="Max solver steps per attempt. Default: 2000.",
    )
    parser.add_argument(
        "--stuck-threshold", type=int, default=40,
        help="Steps without a new pickup before considering the solver stuck. Default: 40.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-level output.",
    )
    args = parser.parse_args(argv)

    levels = _parse_level_range(args.levels)
    generate_all_datasets(
        levels=levels,
        output_dir=Path(args.output_dir),
        max_attempts=args.max_attempts,
        max_steps=args.max_steps,
        stuck_threshold=args.stuck_threshold,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
