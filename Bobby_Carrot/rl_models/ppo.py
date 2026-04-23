"""PPO (Proximal Policy Optimization) agent and training loop.

Implements:
- Shared CNN encoder → Policy head + Value head
- Clipped surrogate objective with entropy bonus
- GAE (Generalised Advantage Estimation)
- Action masking (invalid actions get -inf logits)
- Multi-level curriculum training
- Optional ICM integration
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.optim as optim

from .config import PPOConfig, TrainingConfig, ICMConfig, LevelConfig
from .networks import CNNEncoder, ObservationPreprocessor, PolicyHead, ValueHead
from .buffers import RolloutBuffer


# ---------------------------------------------------------------------------
# P7 helper: running mean/std for return normalization
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford's online algorithm for batch-updated running mean/variance.

    Used to normalize GAE returns so the critic's target scale stays near 1
    across levels with wildly different return magnitudes (L1 ~60 vs L4 ~400).
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = x.size

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-8)))


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent(nn.Module):
    """PPO actor-critic agent with shared CNN backbone and action masking."""

    def __init__(self, config: PPOConfig, n_actions: int = 4) -> None:
        super().__init__()
        self.config = config
        self.n_actions = n_actions

        self.encoder = CNNEncoder(
            channel_sizes=config.cnn_channels,
            hidden_dim=config.hidden_dim,
        )
        self.policy = PolicyHead(config.hidden_dim, n_actions)
        self.value = ValueHead(config.hidden_dim)

    def forward(self, obs: torch.Tensor):
        """Not used directly — use select_action or evaluate_actions."""
        features = self.encoder(obs)
        return self.policy(features), self.value(features)

    @torch.no_grad()
    def select_action(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float, float]:
        """Select action for a single observation with optional masking.

        Returns: (action, log_prob, value)
        """
        features = self.encoder(obs.unsqueeze(0))

        # Pass mask to policy head (will be None if no masking)
        mask = action_mask.unsqueeze(0) if action_mask is not None else None
        dist = self.policy(features, action_mask=mask)
        value = self.value(features)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for a batch of observations.

        Returns: (log_probs, values, entropy)
        """
        features = self.encoder(obs)
        dist = self.policy(features, action_mask=action_masks)
        values = self.value(features)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


# ---------------------------------------------------------------------------
# PPO Training Loop
# ---------------------------------------------------------------------------

def train_ppo(
    ppo_config: PPOConfig,
    train_config: TrainingConfig,
    level_config: LevelConfig,
    icm_config: Optional[ICMConfig] = None,
    resume_path: Optional[str] = None,
) -> PPOAgent:
    """Full PPO training with curriculum learning, action masking, and optional ICM.

    Returns the trained PPOAgent.
    """
    import sys
    _here = Path(__file__).resolve().parent.parent.parent
    game_python = _here / "Game_Python"
    if str(game_python) not in sys.path:
        sys.path.insert(0, str(game_python))
    from bobby_carrot.rl_env import BobbyCarrotEnv  # type: ignore

    # Disable PyTorch distribution argument validation globally.
    # In Python debug mode (__debug__=True, the default), Distribution._validate_args
    # defaults to True and rejects -inf / NaN logits even when validate_args=False is
    # passed to the constructor.  We have our own NaN guards so this is safe to disable.
    torch.distributions.Distribution.set_default_validate_args(False)

    # Device setup
    if train_config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_config.device)
    print(f"[PPO] Using device: {device}")

    # Seed
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Create agent
    agent = PPOAgent(ppo_config).to(device)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt['agent_state_dict'])
        print(f"[PPO] Loaded weights from {resume_path}")
        # Reset policy head for new level distribution to avoid transferred bias
        if train_config.reset_policy_head_on_resume:
            from .networks import init_orthogonal
            agent.policy.linear.reset_parameters()
            init_orthogonal(agent.policy.linear, gain=0.01)
            print("[PPO] Reset policy head for phase transfer (encoder retained)")
    optimizer = optim.Adam(agent.parameters(), lr=ppo_config.lr, eps=1e-5)
    preprocessor = ObservationPreprocessor(device)

    # P5: EMA teacher snapshot for anti-forgetting via KL regularization.
    teacher = PPOAgent(ppo_config).to(device)
    teacher.load_state_dict(agent.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    _teacher_decay = float(train_config.teacher_ema_decay)
    _teacher_kl_coef = float(train_config.teacher_kl_coef)
    _teacher_kl_mastery_coef = float(train_config.teacher_kl_mastery_coef)

    # P7: running normalization of returns for value-loss stability.
    return_rms = RunningMeanStd()

    # Optional ICM
    icm_module = None
    icm_optimizer = None
    if icm_config and icm_config.enabled:
        from .icm import ICMModule
        icm_module = ICMModule(icm_config, ppo_config.hidden_dim).to(device)
        icm_optimizer = optim.Adam(icm_module.parameters(), lr=icm_config.lr)
        print(f"[PPO] ICM enabled (scale={icm_config.intrinsic_reward_scale})")

    # Curriculum
    all_train_levels = list(level_config.train_levels)
    if train_config.curriculum:
        active_levels = all_train_levels[:min(train_config.curriculum_start_levels, len(all_train_levels))]
    else:
        active_levels = all_train_levels

    # Create env for first level
    current_level = active_levels[0]
    env = BobbyCarrotEnv(
        map_kind=current_level[0],
        map_number=current_level[1],
        observation_mode=train_config.observation_mode,
        include_inventory=True,
        headless=True,
        max_steps=train_config.max_steps_per_episode,
    )

    # Determine obs_dim from a reset
    dummy_obs = env.reset()
    obs_dim = len(dummy_obs)

    # Rollout buffer (stores raw int16 obs + action masks)
    rollout = RolloutBuffer(
        rollout_length=ppo_config.rollout_length,
        obs_dim=obs_dim,
        n_actions=agent.n_actions,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
    )

    # Logging
    log_dir = Path(train_config.log_dir) / "ppo"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(train_config.checkpoint_dir) / "ppo"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training_log.csv"
    csv_handle = open(log_file, "w", newline="")
    csv_writer = csv.writer(csv_handle)
    csv_writer.writerow([
        "timestep", "episode", "avg_reward", "avg_success",
        "avg_collected",
        "policy_loss", "value_loss", "entropy", "clip_fraction",
        "active_levels", "elapsed_sec",
    ])

    # Training state
    obs_raw = env.reset()
    obs_tensor = preprocessor(obs_raw)
    action_mask_np = env.get_valid_actions()
    action_mask_tensor = torch.tensor(action_mask_np, dtype=torch.bool, device=device)
    done = False
    total_timesteps = 0
    episode_count = 0
    episode_reward = 0.0
    episode_rewards: List[float] = []
    episode_successes: List[float] = []
    episode_collected_fracs: List[float] = []
    curriculum_window: List[float] = []
    best_avg_success = 0.0
    greedy_gate_streak = 0
    level_cycle_idx = 0  # Round-robin level index for equal exposure
    start_time = time.time()

    # Per-level success tracking for curriculum weighted sampling
    level_success_history: Dict[Tuple[str, int], List[float]] = {
        lvl: [] for lvl in active_levels
    }
    _LEVEL_HISTORY_WINDOW = train_config.level_history_window

    # Anti-forgetting: track per-rollout episode counts per level to enforce
    # a minimum quota for mastered levels, and a dwell counter for promotion.
    promotion_dwell_counter = 0
    last_entropy_boost_until = 0  # total_timesteps at which the boost expires

    # P4 fallback promotion: tracks consecutive windows above the softer
    # fallback threshold when the main promotion threshold isn't met.
    fallback_dwell_counter = 0
    # P6 regression-triggered entropy boost: remember each active level's max
    # observed windowed success; if success drops by >= regression_trigger_drop
    # the entropy boost is re-armed.
    level_success_max: Dict[Tuple[str, int], float] = {}

    print(f"[PPO] Starting training for {train_config.total_timesteps} timesteps")
    print(f"[PPO] Active levels: {len(active_levels)} / {len(all_train_levels)}")
    print(f"[PPO] Observation dim: {obs_dim} | Channels: {preprocessor.num_channels()}")

    while total_timesteps < train_config.total_timesteps:
        # ── Collect rollout ───────────────────────────────────
        rollout.reset()
        agent.eval()

        for _step in range(ppo_config.rollout_length):
            action, log_prob, value = agent.select_action(obs_tensor, action_mask_tensor)

            next_obs_raw, reward, done, info = env.step(action)

            # Scale environmental reward to keep Value Head targets constrained
            reward_scaled = reward * train_config.reward_scale

            # ICM intrinsic reward
            if icm_module is not None:
                next_tensor = preprocessor(next_obs_raw)
                with torch.no_grad():
                    enc_obs = agent.encoder(obs_tensor.unsqueeze(0))
                    enc_next = agent.encoder(next_tensor.unsqueeze(0))
                intrinsic = icm_module.intrinsic_reward(
                    enc_obs, enc_next,
                    torch.tensor([action], device=device),
                )
                reward_scaled += icm_config.intrinsic_reward_scale * intrinsic

            # Store in rollout using the scaled reward
            rollout.add(
                obs_raw.astype(np.int16), action, reward_scaled, done,
                log_prob, value, action_mask_np,
            )
            episode_reward += reward  # Keep tracking raw, unscaled reward for human logs
            total_timesteps += 1

            if done:
                success = 1.0 if info.get("level_completed", False) else 0.0
                episode_rewards.append(episode_reward)
                episode_successes.append(success)
                total_collected = float(info.get("total_collected", 0.0))
                total_targets = float(info.get("total_targets", 0.0))
                collected_frac = (total_collected / total_targets) if total_targets > 0 else 0.0
                episode_collected_fracs.append(collected_frac)
                curriculum_window.append(success)
                if len(curriculum_window) > train_config.curriculum_promotion_window:
                    curriculum_window = curriculum_window[-train_config.curriculum_promotion_window:]
                episode_count += 1
                episode_reward = 0.0

                # Track per-level success
                if current_level not in level_success_history:
                    level_success_history[current_level] = []
                level_success_history[current_level].append(success)
                # Trim history
                if len(level_success_history[current_level]) > _LEVEL_HISTORY_WINDOW * 2:
                    level_success_history[current_level] = level_success_history[current_level][-_LEVEL_HISTORY_WINDOW:]

                # Weighted level sampling: failing levels get more practice.
                # Anti-forgetting: mastered levels get a high floor AND a
                # minimum quota so they cannot be starved when new levels
                # fail hard (the Phase 2 L2/L3 collapse pattern).
                mastery_floor = train_config.curriculum_mastery_floor
                weights = []
                for lvl in active_levels:
                    history = level_success_history.get(lvl, [])
                    if len(history) < 5:
                        # Bug fix: was 2.0 — caused 64.5% sampling spike for new
                        # levels on first addition, flooding rollout with failure
                        # episodes that overwrote shared encoder features.
                        w = 1.0  # Moderate boost for under-explored new levels
                    else:
                        recent_success = float(np.mean(history[-_LEVEL_HISTORY_WINDOW:]))
                        if recent_success >= 0.75:
                            w = mastery_floor  # Mastered — keep practicing
                        elif recent_success >= 0.50:
                            w = max(mastery_floor, 1.0 - recent_success)
                        else:
                            w = max(0.50, 1.0 - recent_success)
                    weights.append(w)
                w_arr = np.array(weights)
                w_sum = w_arr.sum()
                if w_sum > 0:
                    w_arr = w_arr / w_sum
                else:
                    w_arr = np.ones_like(w_arr) / len(w_arr)
                # Enforce minimum quota: no level falls below curriculum_min_quota
                # fraction of sampling mass.  This is the hard anti-forgetting
                # guard on top of the soft mastery floor.
                min_quota = train_config.curriculum_min_quota
                if min_quota > 0 and len(active_levels) > 1:
                    max_quota = 1.0 / len(active_levels)
                    eff_quota = min(min_quota, max_quota * 0.95)
                    deficit_mask = w_arr < eff_quota
                    if deficit_mask.any():
                        needed = (eff_quota - w_arr[deficit_mask]).sum()
                        surplus_mask = w_arr >= eff_quota
                        surplus = w_arr[surplus_mask].sum()
                        if surplus > needed:
                            w_arr[deficit_mask] = eff_quota
                            w_arr[surplus_mask] = w_arr[surplus_mask] * (surplus - needed) / surplus
                            w_sum = w_arr.sum()
                            if w_sum > 0:
                                w_arr = w_arr / w_sum
                            else:
                                w_arr = np.ones_like(w_arr) / len(w_arr)
                # Enforce maximum weight cap: no single level dominates sampling.
                # Without this, a failing level (weight≈1.0) beats two mastered
                # levels (weight≈0.55 each) and gets 47%+ of samples — its
                # failure gradients overwrite shared encoder features for other levels.
                # effective_max is at least 1/N so the cap is always physically enforceable
                # (e.g. with only 2 levels, 50% each is the minimum possible).
                max_level_weight = train_config.curriculum_max_level_weight
                if max_level_weight < 1.0 and len(active_levels) > 1:
                    effective_max = max(max_level_weight, 1.0 / len(active_levels))
                    excess = np.maximum(w_arr - effective_max, 0.0)
                    if excess.sum() > 0:
                        w_arr = np.minimum(w_arr, effective_max)
                        under_mask = w_arr < effective_max
                        if under_mask.any():
                            under_sum = w_arr[under_mask].sum()
                            if under_sum > 0:
                                w_arr[under_mask] += excess.sum() * (w_arr[under_mask] / under_sum)
                            else:
                                w_arr[under_mask] += excess.sum() / under_mask.sum()
                        w_sum = w_arr.sum()
                        if w_sum > 0:
                            w_arr = w_arr / w_sum
                        else:
                            w_arr = np.ones_like(w_arr) / len(w_arr)
                level_cycle_idx = int(np.random.choice(len(active_levels), p=w_arr))
                current_level = active_levels[level_cycle_idx]
                env.set_map(map_kind=current_level[0], map_number=current_level[1])
                obs_raw = env.reset()
                obs_tensor = preprocessor(obs_raw)
                action_mask_np = env.get_valid_actions()
                action_mask_tensor = torch.tensor(action_mask_np, dtype=torch.bool, device=device)
            else:
                obs_raw = next_obs_raw
                obs_tensor = preprocessor(obs_raw)
                action_mask_np = env.get_valid_actions()
                action_mask_tensor = torch.tensor(action_mask_np, dtype=torch.bool, device=device)

        # ── Compute GAE ───────────────────────────────────────
        with torch.no_grad():
            last_tensor = preprocessor(obs_raw)
            last_features = agent.encoder(last_tensor.unsqueeze(0))
            last_value = float(agent.value(last_features).item())

        rollout.compute_gae(last_value, done)

        # P7: update running-return stats from this rollout's returns so the
        # critic learns against a normalized target.
        return_rms.update(rollout.returns[:rollout.ptr])
        return_std = return_rms.std

        # P5: detect whether any active level is "mastered" — if so, strengthen
        # the KL anchor to guard against catastrophic forgetting.
        # Bug fix: threshold lowered 0.75→0.60 so L2 (which peaked at 68%) triggers
        # the stronger KL coef before it starts regressing, not after.
        has_mastered_level = False
        for lvl in active_levels:
            hist = level_success_history.get(lvl, [])
            if len(hist) >= 20:
                if float(np.mean(hist[-_LEVEL_HISTORY_WINDOW:])) >= 0.60:
                    has_mastered_level = True
                    break
        active_kl_coef = _teacher_kl_coef + (
            _teacher_kl_mastery_coef if has_mastered_level else 0.0
        )

        # ── PPO Update ────────────────────────────────────────
        agent.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        total_kl_teacher = 0.0
        update_count = 0

        for _epoch in range(ppo_config.n_epochs):
            for batch in rollout.get_batches(ppo_config.minibatch_size):
                b_obs_raw = batch["observations"]
                b_obs = preprocessor.process_numpy_batch(b_obs_raw.astype(np.int16))
                b_actions = torch.tensor(batch["actions"], dtype=torch.long, device=device)
                b_old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32, device=device)
                b_advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=device)
                b_returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
                b_action_masks = torch.tensor(batch["action_masks"], dtype=torch.bool, device=device)

                if ppo_config.normalize_advantages and b_advantages.numel() > 1:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                # P7: normalize returns the critic regresses against. Division
                # by running std keeps the value head's target near unit scale
                # even as the reward distribution shifts across the curriculum.
                b_returns_norm = b_returns / max(return_std, 1e-3)

                # Shared encoder forward once, then split into heads so both
                # the student *and* the teacher can run through the same
                # features for KL regularization without a second forward.
                features = agent.encoder(b_obs)
                dist = agent.policy(features, action_mask=b_action_masks)
                values = agent.value(features)
                log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy()

                # Clipped surrogate
                ratio = (log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - ppo_config.clip_ratio, 1.0 + ppo_config.clip_ratio) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # P7: Huber loss on normalized returns. Scaled-back variant of
                # the target remains in b_returns for logging, but the critic
                # now learns a unit-scale signal.
                value_loss = nn.functional.huber_loss(
                    values / max(return_std, 1e-3), b_returns_norm
                )

                # P5: KL(student || teacher) anchor. Teacher sees the same
                # features via its own encoder; we compute logits on the raw
                # observations to avoid leaking student features into the
                # teacher path.
                with torch.no_grad():
                    t_features = teacher.encoder(b_obs)
                    t_dist = teacher.policy(t_features, action_mask=b_action_masks)
                    t_log_probs_all = torch.log_softmax(t_dist.logits, dim=-1)
                s_log_probs_all = torch.log_softmax(dist.logits, dim=-1)
                s_probs_all = s_log_probs_all.exp()
                # Mask invalid actions (where probs are 0 and log-probs are -inf)
                # to keep the KL finite and well-defined.
                kl_per_action = s_probs_all * (s_log_probs_all - t_log_probs_all)
                kl_per_action = torch.where(
                    torch.isfinite(kl_per_action),
                    kl_per_action,
                    torch.zeros_like(kl_per_action),
                )
                kl_teacher = kl_per_action.sum(dim=-1).mean()

                # Entropy bonus with linear schedule
                entropy_loss = -entropy.mean()
                # Decay entropy coeff from initial to entropy_min over training
                progress = min(1.0, total_timesteps / train_config.total_timesteps)
                current_entropy_coeff = ppo_config.entropy_coeff + progress * (
                    ppo_config.entropy_min - ppo_config.entropy_coeff
                )
                # Temporary entropy boost after a curriculum promotion —
                # forces exploration on the newly-added level before the
                # schedule collapses entropy.
                if total_timesteps < last_entropy_boost_until:
                    current_entropy_coeff *= train_config.entropy_boost_multiplier

                # Cosine LR decay over the last lr_decay_final_fraction of
                # training so L4/L5 policy settles without re-breaking L1–L3.
                decay_frac = train_config.lr_decay_final_fraction
                if decay_frac > 0:
                    decay_start = 1.0 - decay_frac
                    if progress >= decay_start:
                        lr_prog = (progress - decay_start) / max(1e-8, decay_frac)
                        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, lr_prog)))
                        lr_mult = train_config.lr_decay_min_multiplier + (
                            1.0 - train_config.lr_decay_min_multiplier
                        ) * cosine
                        for pg in optimizer.param_groups:
                            pg["lr"] = ppo_config.lr * lr_mult

                loss = (
                    policy_loss
                    + ppo_config.value_coeff * value_loss
                    + current_entropy_coeff * entropy_loss
                    + active_kl_coef * kl_teacher
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo_config.max_grad_norm)
                # Guard: skip weight update if loss is non-finite to prevent NaN
                # weights from contaminating the encoder on subsequent minibatches.
                if torch.isfinite(loss):
                    optimizer.step()
                else:
                    print(f"[PPO] WARNING: non-finite loss={loss.item():.4f} "
                          f"at t={total_timesteps} — skipping optimizer step")
                    optimizer.zero_grad()

                total_kl_teacher += float(kl_teacher.item()) if torch.isfinite(kl_teacher) else 0.0

                # ICM update — guard against NaN features from a bad PPO step.
                if icm_module is not None and icm_optimizer is not None:
                    enc_obs = agent.encoder(b_obs).detach()
                    if enc_obs.size(0) > 1 and torch.isfinite(enc_obs).all():
                        icm_loss = icm_module.compute_loss(
                            enc_obs[:-1], enc_obs[1:],
                            b_actions[:-1],
                        )
                        if torch.isfinite(icm_loss):
                            icm_optimizer.zero_grad()
                            icm_loss.backward()
                            icm_optimizer.step()

                # Track metrics
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > ppo_config.clip_ratio).float().mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_frac += clip_frac
                update_count += 1

        # EMA-update the teacher once per rollout (not per minibatch).
        # Per-minibatch updates at decay=0.995 let the teacher absorb ~47% of
        # the new policy per rollout — it forgets L_n-1 within 2-3 rollouts of
        # a new level being introduced, breaking the anti-forgetting anchor.
        # Per-rollout at decay=0.99 gives ~1% update per rollout instead.
        with torch.no_grad():
            for t_param, s_param in zip(teacher.parameters(), agent.parameters()):
                t_param.data.mul_(_teacher_decay).add_(
                    s_param.data, alpha=1.0 - _teacher_decay
                )

        # ── Logging ───────────────────────────────────────────
        if total_timesteps % train_config.log_interval < ppo_config.rollout_length:
            avg_reward = float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0
            avg_success = float(np.mean(episode_successes[-50:])) if episode_successes else 0.0
            avg_collected = float(np.mean(episode_collected_fracs[-50:])) if episode_collected_fracs else 0.0
            avg_pl = total_policy_loss / max(1, update_count)
            avg_vl = total_value_loss / max(1, update_count)
            avg_ent = total_entropy / max(1, update_count)
            avg_cf = total_clip_frac / max(1, update_count)
            elapsed = time.time() - start_time

            avg_kl = total_kl_teacher / max(1, update_count)
            print(
                f"[PPO] t={total_timesteps:>7d} | ep={episode_count:>4d} | "
                f"avg_r={avg_reward:>7.2f} | success={avg_success:>5.2%} | "
                f"collect={avg_collected:>5.2%} | "
                f"p_loss={avg_pl:.4f} | v_loss={avg_vl:.4f} | "
                f"ent={avg_ent:.4f} | clip={avg_cf:.3f} | "
                f"kl_t={avg_kl:.4f} | ret_std={return_std:.2f} | "
                f"levels={len(active_levels)} | {elapsed:.0f}s"
            )
            # Per-level success breakdown (critical for diagnosing which levels stall)
            level_parts = []
            for lvl in active_levels:
                history = level_success_history.get(lvl, [])
                if history:
                    recent = history[-_LEVEL_HISTORY_WINDOW:]
                    lvl_success = float(np.mean(recent))
                    level_parts.append(f"{lvl[0][0]}{lvl[1]}={lvl_success:.0%}")
                else:
                    level_parts.append(f"{lvl[0][0]}{lvl[1]}=N/A")
            print(f"[PPO]   per-level: {' | '.join(level_parts)}")
            csv_writer.writerow([
                total_timesteps, episode_count, f"{avg_reward:.4f}", f"{avg_success:.4f}",
                f"{avg_collected:.4f}",
                f"{avg_pl:.6f}", f"{avg_vl:.6f}", f"{avg_ent:.6f}", f"{avg_cf:.4f}",
                len(active_levels), f"{elapsed:.1f}",
            ])
            csv_handle.flush()

        # ── Curriculum Promotion ──────────────────────────────
        # Gate: unlock next level only when the current highest active level
        # has reached threshold success over curriculum_dwell_windows
        # consecutive evaluation windows.  Dwell prevents one lucky window
        # from triggering a premature promotion (Phase 2 L4→L5 pattern).
        # P4: additionally, a *fallback* promotion fires when the highest
        # level has stayed above curriculum_fallback_threshold for
        # curriculum_fallback_windows — so L4/L5 still get training exposure
        # even if L3 plateaus below the main threshold.
        if train_config.curriculum and len(active_levels) < len(all_train_levels):
            highest_active = active_levels[-1]
            history_highest = level_success_history.get(highest_active, [])
            promoted = False
            if len(history_highest) >= _LEVEL_HISTORY_WINDOW:
                highest_success = float(np.mean(history_highest[-_LEVEL_HISTORY_WINDOW:]))
                if highest_success >= train_config.curriculum_promotion_threshold:
                    promotion_dwell_counter += 1
                else:
                    promotion_dwell_counter = 0

                if highest_success >= train_config.curriculum_fallback_threshold:
                    fallback_dwell_counter += 1
                else:
                    fallback_dwell_counter = 0

                main_ready = promotion_dwell_counter >= train_config.curriculum_dwell_windows
                fallback_ready = (
                    fallback_dwell_counter >= train_config.curriculum_fallback_windows
                )
                # Retention gate: block promotion if any non-frontier active level
                # that was previously mastered (≥75%) has regressed below the
                # retention floor. This prevents introducing L_n+1 while L_n is
                # already forgetting — the pattern that caused L2 → 0% collapse.
                retention_floor = train_config.curriculum_retention_floor
                retention_blocked = False
                if retention_floor > 0 and len(active_levels) > 1:
                    for lvl in active_levels[:-1]:
                        hist = level_success_history.get(lvl, [])
                        if len(hist) < _LEVEL_HISTORY_WINDOW:
                            continue
                        recent = float(np.mean(hist[-_LEVEL_HISTORY_WINDOW:]))
                        prev_max = level_success_max.get(lvl, 0.0)
                        if prev_max >= 0.70 and recent < retention_floor:
                            retention_blocked = True
                            print(
                                f"[PPO] Promotion blocked (retention gate): "
                                f"{lvl[0]}{lvl[1]} regressed to {recent:.0%} "
                                f"(was {prev_max:.0%}, floor={retention_floor:.0%})"
                            )
                            break
                if (main_ready or fallback_ready) and not retention_blocked:
                    old_count = len(active_levels)
                    n_add = max(1, train_config.curriculum_add_levels)
                    active_levels = all_train_levels[:min(old_count + n_add, len(all_train_levels))]
                    new_lvl = active_levels[-1]
                    if new_lvl not in level_success_history:
                        level_success_history[new_lvl] = []
                    curriculum_window.clear()
                    promotion_dwell_counter = 0
                    fallback_dwell_counter = 0
                    last_entropy_boost_until = total_timesteps + train_config.entropy_boost_steps
                    trigger = "main" if main_ready else "fallback"
                    print(
                        f"[PPO] Curriculum promotion ({trigger}): {old_count} -> {len(active_levels)} levels "
                        f"({highest_active[0]}{highest_active[1]} success={highest_success:.2%}) "
                        f"| entropy boost active until t={last_entropy_boost_until}"
                    )
                    promoted = True
            if promoted:
                pass  # promotion handled above

        # P6: regression-triggered entropy re-arm.
        # If any active level's rolling success has dropped by ≥
        # regression_trigger_drop from its recorded max, re-arm the entropy
        # boost so exploration can push the policy back off the bad basin.
        regression_detected = False
        for lvl in active_levels:
            hist = level_success_history.get(lvl, [])
            if len(hist) < 10:
                continue
            recent = float(np.mean(hist[-_LEVEL_HISTORY_WINDOW:]))
            prev_max = level_success_max.get(lvl, 0.0)
            if recent > prev_max:
                level_success_max[lvl] = recent
            elif prev_max - recent >= train_config.regression_trigger_drop:
                regression_detected = True
        if regression_detected and total_timesteps >= last_entropy_boost_until:
            last_entropy_boost_until = total_timesteps + train_config.entropy_boost_steps
            print(
                f"[PPO] Regression detected — re-arming entropy boost until "
                f"t={last_entropy_boost_until}"
            )

        # ── Best-model tracking (every rollout) ───────────────
        # Previously this was gated inside the checkpoint_every block (20k
        # steps) which missed the L2 peak at t=110k (60% success) because the
        # next save gate was 10k steps later — by then the policy had already
        # collapsed back to 0%. Check every rollout with a shorter window so
        # transient peaks survive as on-disk checkpoints.
        rolling_window = max(32, min(64, train_config.early_stop_window // 2))
        if (not train_config.greedy_gate_enabled) and len(episode_successes) >= 20:
            recent_success = float(np.mean(episode_successes[-rolling_window:]))
            if recent_success > best_avg_success:
                best_avg_success = recent_success
                best_path = ckpt_dir / "ppo_best.pt"
                torch.save({
                    "agent_state_dict": agent.state_dict(),
                    "total_timesteps": total_timesteps,
                    "episode_count": episode_count,
                    "best_success": best_avg_success,
                }, best_path)
                print(f"[PPO] New best model saved (success={best_avg_success:.2%} over last {rolling_window} eps)")

        # ── Periodic Snapshot Checkpoint ──────────────────────
        if total_timesteps % train_config.checkpoint_every < ppo_config.rollout_length:
            ckpt_path = ckpt_dir / f"ppo_{total_timesteps}.pt"
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "total_timesteps": total_timesteps,
                "episode_count": episode_count,
                "active_levels": active_levels,
                "config": {
                    "ppo": ppo_config,
                    "train": train_config,
                    "level": level_config,
                },
            }, ckpt_path)

        # ── Periodic Evaluation ───────────────────────────────
        if total_timesteps % train_config.eval_interval < ppo_config.rollout_length:
            eval_metrics = _run_eval(
                agent, preprocessor, level_config.test_levels,
                train_config, device, total_timesteps,
            )
            # Anchor "best" to the *eval* distribution too: if the greedy-eval
            # or stochastic-eval success is the strongest signal we've seen,
            # snapshot the agent. This pins best to a policy that actually
            # generalises, not just one that got lucky during training rollouts.
            eval_success = eval_metrics.get("success_rate", 0.0)
            if eval_success > best_avg_success:
                best_avg_success = eval_success
                best_path = ckpt_dir / "ppo_best.pt"
                torch.save({
                    "agent_state_dict": agent.state_dict(),
                    "total_timesteps": total_timesteps,
                    "episode_count": episode_count,
                    "best_success": best_avg_success,
                    "source": "eval",
                }, best_path)
                print(f"[PPO] New best model saved from eval (success={best_avg_success:.2%})")

            # Optional greedy stability gate for single-level reliability runs.
            if train_config.greedy_gate_enabled:
                gate_threshold = float(train_config.greedy_gate_threshold)
                gate_windows = int(train_config.greedy_gate_required_windows)
                if eval_success >= gate_threshold:
                    greedy_gate_streak += 1
                else:
                    greedy_gate_streak = 0
                print(
                    f"[PPO-GATE] greedy={eval_success:.2%} | "
                    f"streak={greedy_gate_streak}/{gate_windows} "
                    f"(threshold={gate_threshold:.0%})"
                )
                if greedy_gate_streak >= gate_windows:
                    best_path = ckpt_dir / "ppo_best.pt"
                    torch.save({
                        "agent_state_dict": agent.state_dict(),
                        "total_timesteps": total_timesteps,
                        "episode_count": episode_count,
                        "best_success": eval_success,
                        "source": "greedy_gate",
                        "greedy_gate_streak": greedy_gate_streak,
                    }, best_path)
                    print(
                        f"[PPO] Greedy gate satisfied: {gate_windows} consecutive "
                        f"windows >= {gate_threshold:.0%}. Saved {best_path} and stopping."
                    )
                    break

        # ── Early Stopping ────────────────────────────────────
        # Single-level demo runs pass early_stop_success > 0 to terminate once
        # the agent has reliably solved its target level. No-op for phased runs.
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
                    f"[PPO] Early-stop: rolling success {recent_success:.1%} "
                    f">= target {train_config.early_stop_success:.1%} "
                    f"over last {train_config.early_stop_window} eps @ t={total_timesteps}"
                )
                # Snapshot best at the stopping point — the current policy is the best we've seen.
                best_path = ckpt_dir / "ppo_best.pt"
                torch.save({
                    "agent_state_dict": agent.state_dict(),
                    "total_timesteps": total_timesteps,
                    "episode_count": episode_count,
                    "best_success": recent_success,
                }, best_path)
                print(f"[PPO] Best model saved to {best_path} (success={recent_success:.2%})")
                break

    # Final save
    final_path = ckpt_dir / "ppo_final.pt"
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "total_timesteps": total_timesteps,
        "episode_count": episode_count,
    }, final_path)
    print(f"[PPO] Training complete. Final model saved to {final_path}")

    csv_handle.close()
    env.close()
    return agent


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _run_eval(
    agent: PPOAgent,
    preprocessor: ObservationPreprocessor,
    test_levels: List[Tuple[str, int]],
    train_config: TrainingConfig,
    device: torch.device,
    timestep: int,
) -> Dict[str, float]:
    """Evaluate on held-out test levels with both greedy and stochastic policies.

    P8: reporting both numbers exposes policies that only *barely* point at the
    right action (greedy succeeds, stochastic regresses) vs. policies that have
    learned a robust distribution (both succeed).  If stochastic success is
    much lower than greedy, the encoder has memorized rather than generalized.
    """
    import sys
    _here = Path(__file__).resolve().parent.parent.parent
    game_python = _here / "Game_Python"
    if str(game_python) not in sys.path:
        sys.path.insert(0, str(game_python))
    from bobby_carrot.rl_env import BobbyCarrotEnv  # type: ignore
    from .evaluate import compute_optimal_path_length

    agent.eval()

    def _run_mode(stochastic: bool) -> Tuple[float, float, float, List[str]]:
        total_success = 0
        total_episodes = 0
        total_reward = 0.0
        path_ratios: List[float] = []
        level_results: List[str] = []
        for kind, num in test_levels:
            env = BobbyCarrotEnv(
                map_kind=kind, map_number=num,
                observation_mode=train_config.observation_mode,
                include_inventory=True, headless=True,
                max_steps=train_config.max_steps_per_episode,
            )
            level_successes = 0
            for _ in range(train_config.eval_episodes_per_level):
                obs_raw = env.reset()
                optimal_len = compute_optimal_path_length(env)
                done = False
                ep_reward = 0.0
                ep_steps = 0
                info: Dict[str, object] = {}
                while not done:
                    obs_t = preprocessor(obs_raw)
                    mask_np = env.get_valid_actions()
                    mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device)
                    with torch.no_grad():
                        features = agent.encoder(obs_t.unsqueeze(0))
                        dist = agent.policy(features, action_mask=mask_t.unsqueeze(0))
                        if stochastic:
                            action = int(dist.sample().item())
                        else:
                            action = int(dist.probs.argmax(dim=-1).item())
                    obs_raw, reward, done, info = env.step(action)
                    ep_reward += reward
                    ep_steps += 1
                total_reward += ep_reward
                if info.get("level_completed", False):
                    total_success += 1
                    level_successes += 1
                    if 0 < optimal_len < 10**9:
                        path_ratios.append(ep_steps / float(optimal_len))
                total_episodes += 1
            env.close()
            level_results.append(f"{kind}-{num}:{level_successes}/{train_config.eval_episodes_per_level}")
        avg_success = total_success / max(1, total_episodes)
        avg_reward = total_reward / max(1, total_episodes)
        # Step efficiency: mean(actual_steps / optimal_steps) across successful episodes.
        # 1.0 = optimal, higher = wasted steps (e.g. L1 wander). Reported as NaN if no successes.
        avg_ratio = float(np.mean(path_ratios)) if path_ratios else float("nan")
        return avg_success, avg_reward, avg_ratio, level_results

    greedy_success, greedy_reward, greedy_ratio, greedy_levels = _run_mode(stochastic=False)
    stoch_success, stoch_reward, stoch_ratio, _stoch_levels = _run_mode(stochastic=True)

    def _fmt_ratio(r: float) -> str:
        return f"{r:.2f}x" if not (r != r) else "—"  # NaN check

    print(
        f"[PPO-EVAL] t={timestep} | "
        f"greedy={greedy_success:.2%} (r={greedy_reward:.2f}, steps/opt={_fmt_ratio(greedy_ratio)}) | "
        f"stoch={stoch_success:.2%} (r={stoch_reward:.2f}, steps/opt={_fmt_ratio(stoch_ratio)}) | "
        f"{', '.join(greedy_levels)}"
    )
    return {
        "success_rate": greedy_success,
        "avg_reward": greedy_reward,
        "stoch_success_rate": stoch_success,
        "stoch_avg_reward": stoch_reward,
        "greedy_step_ratio": greedy_ratio,
        "stoch_step_ratio": stoch_ratio,
    }
