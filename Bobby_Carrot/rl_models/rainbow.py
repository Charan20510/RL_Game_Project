"""Rainbow DQN agent and training loop.

Combines all 6 DQN enhancements:
1. Double DQN — online selects, target evaluates
2. Dueling Architecture — separate V(s) and A(s,a) streams
3. Prioritized Experience Replay (PER) — priority-based sampling
4. NoisyNet — parameter-space exploration (no ε-greedy)
5. N-step Returns — multi-step bootstrapped targets
6. C51 (Distributional) — categorical distribution over returns
"""

from __future__ import annotations

import copy
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import RainbowConfig, TrainingConfig, ICMConfig, LevelConfig
from .networks import CNNEncoder, ObservationPreprocessor, DuelingDistributionalHead
from .buffers import PrioritizedReplayBuffer, NStepReplayBuffer


# ---------------------------------------------------------------------------
# Rainbow DQN Agent
# ---------------------------------------------------------------------------

class RainbowAgent(nn.Module):
    """Rainbow DQN agent: CNN encoder + Dueling Distributional head with NoisyNet."""

    def __init__(self, config: RainbowConfig, n_actions: int = 4) -> None:
        super().__init__()
        self.config = config
        self.n_actions = n_actions

        self.encoder = CNNEncoder(
            channel_sizes=config.cnn_channels,
            hidden_dim=config.hidden_dim,
        )
        self.head = DuelingDistributionalHead(
            input_dim=config.hidden_dim,
            n_actions=n_actions,
            atom_size=config.atom_size,
            v_min=config.v_min,
            v_max=config.v_max,
            hidden_dim=config.hidden_dim,
            noisy_std=config.noisy_std,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities over atoms for each action.

        Shape: (B, n_actions, atom_size)
        """
        features = self.encoder(obs)
        return self.head(features)

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns expected Q-values. Shape: (B, n_actions)."""
        features = self.encoder(obs)
        return self.head.q_values(features)

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> int:
        """Select greedy action from distributional Q-values (no ε needed — NoisyNet explores)."""
        q = self.q_values(obs.unsqueeze(0))
        return int(q.argmax(dim=-1).item())

    def reset_noise(self) -> None:
        """Resample noise in all NoisyLinear layers."""
        self.head.reset_noise()


# ---------------------------------------------------------------------------
# Rainbow DQN Training Loop
# ---------------------------------------------------------------------------

def train_rainbow(
    rainbow_config: RainbowConfig,
    train_config: TrainingConfig,
    level_config: LevelConfig,
    icm_config: Optional[ICMConfig] = None,
) -> RainbowAgent:
    """Full Rainbow DQN training loop.

    Returns the trained RainbowAgent.
    """
    import sys
    _here = Path(__file__).resolve().parent.parent.parent
    game_python = _here / "Game_Python"
    if str(game_python) not in sys.path:
        sys.path.insert(0, str(game_python))
    from bobby_carrot.rl_env import BobbyCarrotEnv  # type: ignore

    # Device setup
    if train_config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_config.device)
    print(f"[Rainbow] Using device: {device}")

    # Seed
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Create networks
    online_net = RainbowAgent(rainbow_config).to(device)
    target_net = copy.deepcopy(online_net).to(device)
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(online_net.parameters(), lr=rainbow_config.lr, eps=1.5e-4)
    preprocessor = ObservationPreprocessor(device)

    # Optional ICM
    icm_module = None
    icm_optimizer = None
    if icm_config and icm_config.enabled:
        from .icm import ICMModule
        icm_module = ICMModule(icm_config, rainbow_config.hidden_dim).to(device)
        icm_optimizer = optim.Adam(icm_module.parameters(), lr=icm_config.lr)

    # Curriculum
    all_train_levels = list(level_config.train_levels)
    if train_config.curriculum:
        active_levels = all_train_levels[:train_config.curriculum_start_levels]
    else:
        active_levels = all_train_levels

    # Create env
    current_level = active_levels[0]
    env = BobbyCarrotEnv(
        map_kind=current_level[0],
        map_number=current_level[1],
        observation_mode=train_config.observation_mode,
        include_inventory=True,
        headless=True,
        max_steps=train_config.max_steps_per_episode,
    )

    dummy_obs = env.reset()
    obs_dim = len(dummy_obs)

    # Create replay buffer with n-step wrapper
    per_buffer = PrioritizedReplayBuffer(
        capacity=rainbow_config.buffer_size,
        obs_dim=obs_dim,
        alpha=rainbow_config.per_alpha,
        beta_start=rainbow_config.per_beta_start,
        beta_end=rainbow_config.per_beta_end,
        beta_anneal_steps=rainbow_config.per_beta_anneal_steps,
        epsilon=rainbow_config.per_epsilon,
    )
    replay = NStepReplayBuffer(
        per_buffer=per_buffer,
        n_step=rainbow_config.n_step,
        gamma=rainbow_config.gamma,
    )

    # Support atoms for distributional RL
    support = torch.linspace(
        rainbow_config.v_min, rainbow_config.v_max, rainbow_config.atom_size,
    ).to(device)
    delta_z = (rainbow_config.v_max - rainbow_config.v_min) / (rainbow_config.atom_size - 1)

    # Logging
    log_dir = Path(train_config.log_dir) / "rainbow"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(train_config.checkpoint_dir) / "rainbow"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training_log.csv"
    csv_handle = open(log_file, "w", newline="")
    csv_writer = csv.writer(csv_handle)
    csv_writer.writerow([
        "timestep", "episode", "avg_reward", "avg_success",
        "loss", "avg_q", "buffer_size", "beta",
        "active_levels", "elapsed_sec",
    ])

    # Training state
    obs_raw = env.reset()
    total_timesteps = 0
    episode_count = 0
    episode_reward = 0.0
    episode_rewards: List[float] = []
    episode_successes: List[float] = []
    curriculum_window: List[float] = []
    losses: List[float] = []
    q_values_log: List[float] = []
    best_rolling_success = 0.0
    start_time = time.time()

    print(f"[Rainbow] Starting training for {train_config.total_timesteps} timesteps")
    print(f"[Rainbow] Learning starts at {rainbow_config.learning_starts} steps")
    print(f"[Rainbow] Active levels: {len(active_levels)} / {len(all_train_levels)}")

    while total_timesteps < train_config.total_timesteps:
        # ── Select Action ─────────────────────────────────────
        obs_tensor = preprocessor(obs_raw)
        online_net.reset_noise()

        if total_timesteps < rainbow_config.learning_starts:
            action = np.random.randint(0, online_net.n_actions)
        else:
            online_net.eval()
            action = online_net.select_action(obs_tensor)

        # ── Step Environment ──────────────────────────────────
        next_obs_raw, reward, done, info = env.step(action)

        # ICM intrinsic reward
        if icm_module is not None and total_timesteps >= rainbow_config.learning_starts:
            next_tensor = preprocessor(next_obs_raw)
            with torch.no_grad():
                enc_obs = online_net.encoder(obs_tensor.unsqueeze(0))
                enc_next = online_net.encoder(next_tensor.unsqueeze(0))
            intrinsic = icm_module.intrinsic_reward(
                enc_obs, enc_next,
                torch.tensor([action], device=device),
            )
            reward += icm_config.intrinsic_reward_scale * intrinsic

        # Store in replay
        replay.add(
            obs_raw.astype(np.float32), action, reward,
            next_obs_raw.astype(np.float32), done,
        )
        episode_reward += reward
        total_timesteps += 1

        if done:
            success = 1.0 if info.get("level_completed", False) else 0.0
            episode_rewards.append(episode_reward)
            episode_successes.append(success)
            curriculum_window.append(success)
            if len(curriculum_window) > train_config.curriculum_promotion_window:
                curriculum_window = curriculum_window[-train_config.curriculum_promotion_window:]
            episode_count += 1
            episode_reward = 0.0

            # Switch to random level
            current_level = active_levels[np.random.randint(len(active_levels))]
            env.set_map(map_kind=current_level[0], map_number=current_level[1])
            obs_raw = env.reset()
        else:
            obs_raw = next_obs_raw

        # ── Learn ─────────────────────────────────────────────
        if total_timesteps >= rainbow_config.learning_starts and len(replay) >= rainbow_config.batch_size:
            online_net.train()
            online_net.reset_noise()
            target_net.reset_noise()

            batch, tree_indices, is_weights = replay.sample(rainbow_config.batch_size)
            is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=device)

            # Process observations
            b_obs = preprocessor.process_numpy_batch(batch["obs"])
            b_next_obs = preprocessor.process_numpy_batch(batch["next_obs"])
            b_actions = torch.tensor(batch["actions"], dtype=torch.long, device=device)
            b_rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
            b_dones = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

            # ── Distributional Bellman Target ─────────────────
            with torch.no_grad():
                # Double DQN: online net selects actions, target net evaluates
                next_q_online = online_net.q_values(b_next_obs)
                next_actions = next_q_online.argmax(dim=-1)  # (B,)

                # Get target distribution for selected actions
                target_log_probs = target_net(b_next_obs)  # (B, A, atoms)
                next_actions_expand = next_actions.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, 1, rainbow_config.atom_size
                )
                target_dist = target_log_probs.exp().gather(1, next_actions_expand).squeeze(1)  # (B, atoms)

                # Categorical projection of Bellman target
                tz = b_rewards.unsqueeze(-1) + (1.0 - b_dones.unsqueeze(-1)) * (
                    rainbow_config.gamma ** rainbow_config.n_step
                ) * support.unsqueeze(0)
                tz = tz.clamp(rainbow_config.v_min, rainbow_config.v_max)

                b_idx = ((tz - rainbow_config.v_min) / delta_z)
                lower = b_idx.floor().long()
                upper = b_idx.ceil().long()

                # Clamp to valid range
                lower = lower.clamp(0, rainbow_config.atom_size - 1)
                upper = upper.clamp(0, rainbow_config.atom_size - 1)

                # Distribute probability mass
                projected = torch.zeros_like(target_dist)
                offset = torch.linspace(
                    0, (rainbow_config.batch_size - 1) * rainbow_config.atom_size,
                    rainbow_config.batch_size,
                ).long().unsqueeze(1).to(device)

                projected.view(-1).index_add_(
                    0, (lower + offset).view(-1),
                    (target_dist * (upper.float() - b_idx)).view(-1),
                )
                projected.view(-1).index_add_(
                    0, (upper + offset).view(-1),
                    (target_dist * (b_idx - lower.float())).view(-1),
                )

            # ── Online network log-probs for taken actions ────
            log_probs = online_net(b_obs)  # (B, A, atoms)
            actions_expand = b_actions.unsqueeze(-1).unsqueeze(-1).expand(
                -1, 1, rainbow_config.atom_size
            )
            chosen_log_probs = log_probs.gather(1, actions_expand).squeeze(1)  # (B, atoms)

            # Cross-entropy loss with IS weights
            elementwise_loss = -(projected * chosen_log_probs).sum(dim=-1)
            loss = (is_weights_t * elementwise_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), rainbow_config.max_grad_norm)
            optimizer.step()

            # Update priorities
            td_errors = elementwise_loss.detach().cpu().numpy()
            replay.update_priorities(tree_indices, td_errors)

            losses.append(loss.item())
            with torch.no_grad():
                avg_q = online_net.q_values(b_obs).max(dim=-1).values.mean().item()
            q_values_log.append(avg_q)

            # ICM update
            if icm_module is not None and icm_optimizer is not None:
                enc_obs = online_net.encoder(b_obs).detach()
                enc_next = online_net.encoder(b_next_obs).detach()
                icm_loss = icm_module.compute_loss(enc_obs, enc_next, b_actions)
                icm_optimizer.zero_grad()
                icm_loss.backward()
                icm_optimizer.step()

            # ── Target Network Update ─────────────────────────
            if total_timesteps % rainbow_config.target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

        # ── Logging ───────────────────────────────────────────
        if total_timesteps % train_config.log_interval == 0 and total_timesteps > 0:
            avg_reward = float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0
            avg_success = float(np.mean(episode_successes[-50:])) if episode_successes else 0.0
            avg_loss = float(np.mean(losses[-100:])) if losses else 0.0
            avg_q = float(np.mean(q_values_log[-100:])) if q_values_log else 0.0
            elapsed = time.time() - start_time

            print(
                f"[Rainbow] t={total_timesteps:>7d} | ep={episode_count:>4d} | "
                f"avg_r={avg_reward:>7.2f} | success={avg_success:>5.2%} | "
                f"loss={avg_loss:.4f} | Q={avg_q:.2f} | "
                f"buf={len(replay)} | beta={per_buffer.beta:.3f} | "
                f"levels={len(active_levels)} | {elapsed:.0f}s"
            )
            csv_writer.writerow([
                total_timesteps, episode_count, f"{avg_reward:.4f}", f"{avg_success:.4f}",
                f"{avg_loss:.6f}", f"{avg_q:.4f}", len(replay), f"{per_buffer.beta:.4f}",
                len(active_levels), f"{elapsed:.1f}",
            ])
            csv_handle.flush()

        # ── Best-model tracking (every episode boundary) ─────
        rolling_window = max(20, min(train_config.early_stop_window, 50))
        if len(episode_successes) >= rolling_window:
            recent_success = float(np.mean(episode_successes[-rolling_window:]))
            if recent_success > best_rolling_success:
                best_rolling_success = recent_success
                best_path = ckpt_dir / "rainbow_best.pt"
                torch.save({
                    "online_state_dict": online_net.state_dict(),
                    "total_timesteps": total_timesteps,
                    "episode_count": episode_count,
                    "best_success": best_rolling_success,
                }, best_path)
                print(
                    f"[Rainbow] New best model saved "
                    f"(success={best_rolling_success:.2%} over last {rolling_window} eps)"
                )

        # ── Curriculum Promotion ──────────────────────────────
        if (
            train_config.curriculum
            and len(curriculum_window) >= train_config.curriculum_promotion_window
            and total_timesteps % 1000 == 0
        ):
            rolling_success = float(np.mean(curriculum_window))
            if (
                rolling_success >= train_config.curriculum_promotion_threshold
                and len(active_levels) < len(all_train_levels)
            ):
                n_add = min(train_config.curriculum_add_levels, len(all_train_levels) - len(active_levels))
                old_count = len(active_levels)
                active_levels = all_train_levels[:old_count + n_add]
                curriculum_window.clear()
                print(
                    f"[Rainbow] Curriculum promotion: {old_count} -> {len(active_levels)} levels "
                    f"(rolling_success={rolling_success:.2%})"
                )

        # ── Checkpointing ─────────────────────────────────────
        if total_timesteps % train_config.checkpoint_every == 0 and total_timesteps > 0:
            ckpt_path = ckpt_dir / f"rainbow_{total_timesteps}.pt"
            torch.save({
                "online_state_dict": online_net.state_dict(),
                "target_state_dict": target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "total_timesteps": total_timesteps,
                "episode_count": episode_count,
                "active_levels": active_levels,
            }, ckpt_path)

        # ── Periodic Evaluation ───────────────────────────────
        if total_timesteps % train_config.eval_interval == 0 and total_timesteps > 0:
            eval_result = _run_rainbow_eval(
                online_net, preprocessor, level_config.test_levels,
                train_config, total_timesteps,
            )
            # Anchor best-model to eval success so the saved checkpoint
            # reflects the policy that generalises, not just a training peak.
            eval_success = eval_result.get("success_rate", 0.0)
            if eval_success > best_rolling_success:
                best_rolling_success = eval_success
                best_path = ckpt_dir / "rainbow_best.pt"
                torch.save({
                    "online_state_dict": online_net.state_dict(),
                    "total_timesteps": total_timesteps,
                    "episode_count": episode_count,
                    "best_success": best_rolling_success,
                    "source": "eval",
                }, best_path)
                print(
                    f"[Rainbow] New best model saved from eval "
                    f"(success={best_rolling_success:.2%})"
                )

        # ── Early Stopping ────────────────────────────────────
        if (
            train_config.early_stop_success > 0.0
            and total_timesteps >= train_config.early_stop_min_timesteps
            and len(episode_successes) >= train_config.early_stop_window
        ):
            recent_success = float(
                np.mean(episode_successes[-train_config.early_stop_window:])
            )
            if recent_success >= train_config.early_stop_success:
                print(
                    f"[Rainbow] Early-stop: rolling success {recent_success:.1%} "
                    f">= target {train_config.early_stop_success:.1%} "
                    f"over last {train_config.early_stop_window} eps @ t={total_timesteps}"
                )
                best_path = ckpt_dir / "rainbow_best.pt"
                torch.save({
                    "online_state_dict": online_net.state_dict(),
                    "total_timesteps": total_timesteps,
                    "episode_count": episode_count,
                    "best_success": recent_success,
                }, best_path)
                print(f"[Rainbow] Best model saved to {best_path} (success={recent_success:.2%})")
                break

    # Final save
    final_path = ckpt_dir / "rainbow_final.pt"
    torch.save({
        "online_state_dict": online_net.state_dict(),
        "total_timesteps": total_timesteps,
        "episode_count": episode_count,
    }, final_path)
    print(f"[Rainbow] Training complete. Final model saved to {final_path}")

    csv_handle.close()
    env.close()
    return online_net


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _run_rainbow_eval(
    agent: RainbowAgent,
    preprocessor: ObservationPreprocessor,
    test_levels: List[Tuple[str, int]],
    train_config: TrainingConfig,
    timestep: int,
) -> Dict[str, float]:
    """Run deterministic evaluation on test levels."""
    import sys
    _here = Path(__file__).resolve().parent.parent.parent
    game_python = _here / "Game_Python"
    if str(game_python) not in sys.path:
        sys.path.insert(0, str(game_python))
    from bobby_carrot.rl_env import BobbyCarrotEnv  # type: ignore

    agent.eval()
    total_success = 0
    total_episodes = 0
    total_reward = 0.0

    for kind, num in test_levels:
        env = BobbyCarrotEnv(
            map_kind=kind, map_number=num,
            observation_mode=train_config.observation_mode,
            include_inventory=True, headless=True,
            max_steps=train_config.max_steps_per_episode,
        )
        for _ in range(train_config.eval_episodes_per_level):
            obs_raw = env.reset()
            done = False
            ep_reward = 0.0
            info = {}
            while not done:
                obs_t = preprocessor(obs_raw)
                with torch.no_grad():
                    q = agent.q_values(obs_t.unsqueeze(0))
                    action = int(q.argmax(dim=-1).item())
                obs_raw, reward, done, info = env.step(action)
                ep_reward += reward
            total_reward += ep_reward
            if info.get("level_completed", False):
                total_success += 1
            total_episodes += 1
        env.close()

    avg_success = total_success / max(1, total_episodes)
    avg_reward = total_reward / max(1, total_episodes)
    print(
        f"[Rainbow-EVAL] t={timestep} | test_success={avg_success:.2%} "
        f"| test_reward={avg_reward:.2f} | episodes={total_episodes}"
    )
    return {"success_rate": avg_success, "avg_reward": avg_reward}
