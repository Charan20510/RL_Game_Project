"""Behavioral Cloning (BC) pre-training for Bobby Carrot.

Loads expert trajectory datasets and trains the PPOAgent encoder + policy head
via supervised cross-entropy loss to establish a strong navigation baseline
before RL fine-tuning.

Two modes:
  per-level  — one dedicated checkpoint per level (default).
  joint      — single model trained on all levels together.

Usage:
    # Generate datasets first:
    python -m Bobby_Carrot.rl_models.dataset_gen

    # Per-level BC (one checkpoint per level)
    python -m Bobby_Carrot.rl_models.bc_pretrain --data-dir expert_data --ckpt-dir checkpoints/bc

    # Joint BC on all levels at once
    python -m Bobby_Carrot.rl_models.bc_pretrain --data-dir expert_data --ckpt-dir checkpoints/bc --joint

    # Load a pre-existing BC checkpoint as starting point
    python -m Bobby_Carrot.rl_models.bc_pretrain --resume checkpoints/bc/normal_01_bc.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent.parent
_GAME_PYTHON = _HERE / "Game_Python"
if str(_GAME_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GAME_PYTHON))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from Bobby_Carrot.rl_models.config import PPOConfig
from Bobby_Carrot.rl_models.networks import CNNEncoder, ObservationPreprocessor, PolicyHead, ValueHead
from Bobby_Carrot.rl_models.ppo import PPOAgent
from Bobby_Carrot.rl_models.dataset_gen import load_dataset


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class ExpertDataset(Dataset):
    """Wraps (observations, actions) arrays as a torch Dataset."""

    def __init__(
        self,
        observations: np.ndarray,   # (N, obs_dim) int16
        actions: np.ndarray,         # (N,)         int8
        preprocessor: ObservationPreprocessor,
    ) -> None:
        assert len(observations) == len(actions), "obs/action length mismatch"
        self.observations = observations
        self.actions = torch.from_numpy(actions.astype(np.int64))
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self.preprocessor.process_single(self.observations[idx])
        return obs_tensor, self.actions[idx]


# ---------------------------------------------------------------------------
# BC training for one dataset
# ---------------------------------------------------------------------------

def train_bc_on_dataset(
    agent: PPOAgent,
    dataset: ExpertDataset,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> List[float]:
    """Train the agent's encoder + policy head via cross-entropy on one dataset.

    The value head is NOT updated during BC (it has no supervised signal).
    Returns a list of per-epoch cross-entropy losses.
    """
    # Only optimize encoder + policy head
    bc_params = list(agent.encoder.parameters()) + list(agent.policy.parameters())
    optimizer = optim.Adam(bc_params, lr=lr, eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss()

    epoch_losses: List[float] = []
    agent.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            features = agent.encoder(obs_batch)
            logits = agent.policy.linear(features)       # raw logits before masking

            loss = criterion(logits, act_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(bc_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(act_batch)
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == act_batch).sum().item())
            total_samples += len(act_batch)

        scheduler.step()
        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        epoch_losses.append(avg_loss)

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(
                f"    epoch {epoch:3d}/{epochs} | "
                f"loss={avg_loss:.4f} | acc={accuracy:.1%}"
            )

    return epoch_losses


# ---------------------------------------------------------------------------
# Per-level BC training
# ---------------------------------------------------------------------------

def train_bc_level(
    map_kind: str,
    map_number: int,
    data_dir: Path,
    ckpt_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    resume_from: Optional[Path] = None,
    ppo_config: Optional[PPOConfig] = None,
    verbose: bool = True,
) -> Optional[Path]:
    """BC pre-train on one level's expert data.  Returns checkpoint path or None."""
    level_tag = f"{map_kind}_{map_number:02d}"
    data_path = data_dir / f"{level_tag}.npz"
    ckpt_path = ckpt_dir / f"{level_tag}_bc.pt"

    if not data_path.exists():
        if verbose:
            print(f"  [{level_tag}] No dataset at {data_path}, skipping BC.")
        return None

    observations, actions = load_dataset(data_path)
    if verbose:
        print(
            f"\n  [{level_tag}] BC training | N={len(actions)} steps | "
            f"epochs={epochs} | lr={lr}"
        )

    cfg = ppo_config or PPOConfig()
    preprocessor = ObservationPreprocessor(device)
    agent = PPOAgent(cfg).to(device)

    # Optionally resume from a previous checkpoint (encoder warm-start)
    if resume_from and resume_from.exists():
        ckpt = torch.load(str(resume_from), map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        if verbose:
            print(f"    Resumed encoder from {resume_from.name}")

    dataset = ExpertDataset(observations, actions, preprocessor)
    epoch_losses = train_bc_on_dataset(
        agent, dataset,
        epochs=epochs, batch_size=batch_size, lr=lr,
        device=device, verbose=verbose,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "level_kind": map_kind,
            "level_num": map_number,
            "phase": "bc",
            "epochs": epochs,
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
            "ppo_config": cfg,
        },
        str(ckpt_path),
    )
    if verbose:
        print(f"    Saved BC checkpoint → {ckpt_path}")

    return ckpt_path


# ---------------------------------------------------------------------------
# Joint BC training across all levels
# ---------------------------------------------------------------------------

def train_bc_joint(
    levels: List[Tuple[str, int]],
    data_dir: Path,
    ckpt_dir: Path,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    resume_from: Optional[Path] = None,
    ppo_config: Optional[PPOConfig] = None,
    verbose: bool = True,
) -> Optional[Path]:
    """Train ONE model jointly on all available expert datasets.

    Useful as a warm-start for per-level RL training that needs good
    generalisation across all normal levels from the start.
    Returns the path to the saved joint checkpoint.
    """
    preprocessor = ObservationPreprocessor(device)
    all_obs: List[np.ndarray] = []
    all_acts: List[np.ndarray] = []

    for kind, num in levels:
        level_tag = f"{kind}_{num:02d}"
        data_path = data_dir / f"{level_tag}.npz"
        if data_path.exists():
            obs, acts = load_dataset(data_path)
            all_obs.append(obs)
            all_acts.append(acts.astype(np.int8))

    if not all_obs:
        print("[BC-Joint] No datasets found — aborting.")
        return None

    combined_obs = np.concatenate(all_obs, axis=0)
    combined_acts = np.concatenate(all_acts, axis=0)
    if verbose:
        print(
            f"\n[BC-Joint] Loaded {len(combined_acts)} transitions across "
            f"{len(all_obs)}/{len(levels)} levels."
        )

    cfg = ppo_config or PPOConfig()
    agent = PPOAgent(cfg).to(device)

    if resume_from and resume_from.exists():
        ckpt = torch.load(str(resume_from), map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        if verbose:
            print(f"  Resumed from {resume_from.name}")

    dataset = ExpertDataset(combined_obs, combined_acts, preprocessor)
    epoch_losses = train_bc_on_dataset(
        agent, dataset,
        epochs=epochs, batch_size=batch_size, lr=lr,
        device=device, verbose=verbose,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "joint_bc.pt"
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "level_kind": "all",
            "level_num": -1,
            "phase": "bc_joint",
            "epochs": epochs,
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
            "ppo_config": cfg,
        },
        str(ckpt_path),
    )
    if verbose:
        print(f"\n[BC-Joint] Saved joint checkpoint → {ckpt_path}")

    return ckpt_path


# ---------------------------------------------------------------------------
# Batch per-level BC
# ---------------------------------------------------------------------------

def train_bc_all_levels(
    levels: List[Tuple[str, int]],
    data_dir: Path,
    ckpt_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    resume_from: Optional[Path] = None,
    ppo_config: Optional[PPOConfig] = None,
    verbose: bool = True,
) -> Dict[Tuple[str, int], Optional[Path]]:
    """Run per-level BC for every level that has a dataset file.

    Each level gets its own checkpoint file named `<kind>_<num>_bc.pt`.
    Returns a dict mapping (kind, num) → checkpoint path (or None if skipped).
    """
    results: Dict[Tuple[str, int], Optional[Path]] = {}
    t0 = time.time()
    print(f"\n[BC] Starting per-level behavioral cloning for {len(levels)} levels.")

    for kind, num in levels:
        ckpt = train_bc_level(
            map_kind=kind,
            map_number=num,
            data_dir=data_dir,
            ckpt_dir=ckpt_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            resume_from=resume_from,
            ppo_config=ppo_config,
            verbose=verbose,
        )
        results[(kind, num)] = ckpt

    saved = sum(1 for v in results.values() if v is not None)
    print(
        f"\n[BC] Done in {time.time() - t0:.1f}s — "
        f"{saved}/{len(levels)} BC checkpoints saved to {ckpt_dir}."
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning pre-training for Bobby Carrot RL agent."
    )
    parser.add_argument(
        "--data-dir", default="expert_data",
        help="Directory containing .npz expert datasets. Default: expert_data/",
    )
    parser.add_argument(
        "--ckpt-dir", default="checkpoints/bc",
        help="Directory to write BC checkpoints. Default: checkpoints/bc/",
    )
    parser.add_argument(
        "--levels", default="1-30",
        help="Levels to train on (e.g. '1-30'). Default: 1-30.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs per level. Default: 50.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size. Default: 64.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Adam learning rate. Default: 3e-4.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="'cpu', 'cuda', or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--joint", action="store_true",
        help="Train one model jointly on all levels instead of per-level.",
    )
    parser.add_argument(
        "--joint-epochs", type=int, default=100,
        help="Epochs for joint BC training. Default: 100.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to existing checkpoint to warm-start from.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-epoch output.",
    )
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[BC] Using device: {device}")

    levels = _parse_level_range(args.levels)
    resume = Path(args.resume) if args.resume else None

    if args.joint:
        train_bc_joint(
            levels=levels,
            data_dir=Path(args.data_dir),
            ckpt_dir=Path(args.ckpt_dir),
            epochs=args.joint_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            resume_from=resume,
            verbose=not args.quiet,
        )
    else:
        train_bc_all_levels(
            levels=levels,
            data_dir=Path(args.data_dir),
            ckpt_dir=Path(args.ckpt_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            resume_from=resume,
            verbose=not args.quiet,
        )


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


if __name__ == "__main__":
    main()
