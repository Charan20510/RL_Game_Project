from __future__ import annotations

# pyright: reportMissingImports=false

import pickle
import sys
import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Allow imports without requiring editable install.
_HERE = Path(__file__).resolve()
ROOT = _HERE.parent
while not (ROOT / "Game_Python").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

GAME_PYTHON_DIR = ROOT / "Game_Python"
if not GAME_PYTHON_DIR.exists():
    raise RuntimeError("Could not locate Game_Python directory for imports.")

if str(GAME_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_PYTHON_DIR))

from bobby_carrot.rl_env import BobbyCarrotEnv  # type: ignore[missing-import]


@dataclass
class QLearningConfig:
    episodes: int = 10000
    alpha: float = 0.15
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.998
    max_steps: int = 500
    report_every: int = 50
    preview_every: int = 0
    curriculum: bool = True
    curriculum_max_level: int = 5
    curriculum_step_episodes: int = 1500
    curriculum_adaptive: bool = True
    curriculum_promotion_window: int = 100
    curriculum_promotion_success: float = 0.9
    curriculum_level_up_epsilon: float = 0.5
    debug_env: bool = False
    debug_every: int = 100
    model_path: Path = Path(__file__).resolve().parent / "q_table_bobby.pkl"


def _obs_key(env: BobbyCarrotEnv, obs: np.ndarray) -> Tuple[int, ...]:
    return env.observation_to_key(obs)


def _select_greedy_action(
    q_table: Dict[Any, np.ndarray],
    state_key: Tuple[int, ...],
    action_space_n: int,
) -> int:
    # Support both tuple keys and legacy byte-encoded keys from old checkpoints.
    q_values = q_table.get(state_key)
    if q_values is None:
        key_bytes = np.asarray(state_key, dtype=np.int16).tobytes()
        q_values = q_table.get(key_bytes)

    if q_values is None:
        return int(np.random.randint(0, action_space_n))

    q_arr = np.asarray(q_values, dtype=np.float32)
    usable = q_arr[:action_space_n]
    if usable.size == 0:
        return int(np.random.randint(0, action_space_n))
    return int(np.argmax(usable))


def _epsilon_greedy_action(
    q_table: Dict[Any, np.ndarray],
    state_key: Tuple[int, ...],
    action_space_n: int,
    epsilon: float,
) -> int:
    if np.random.random() < epsilon:
        return int(np.random.randint(0, action_space_n))

    if state_key not in q_table:
        q_table[state_key] = np.zeros(action_space_n, dtype=np.float32)
    return _select_greedy_action(q_table=q_table, state_key=state_key, action_space_n=action_space_n)


def train_q_learning(
    map_kind: str = "normal",
    map_number: int = 1,
    observation_mode: str = "compact",
    local_view_size: int = 3,
    config: QLearningConfig | None = None,
) -> Dict[Tuple[int, ...], np.ndarray]:
    cfg = config or QLearningConfig()

    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=map_number,
        observation_mode=observation_mode,
        local_view_size=local_view_size,
        include_inventory=True,
        headless=True,
        max_steps=cfg.max_steps,
        debug=cfg.debug_env,
        debug_every=cfg.debug_every,
    )

    q_table: Dict[Tuple[int, ...], np.ndarray] = {}
    epsilon = cfg.epsilon_start

    reward_history: List[float] = []
    success_history: List[float] = []
    all_collected_history: List[float] = []
    step_history: List[int] = []

    current_level = map_number
    level_episode_count = 0
    level_success_window = deque(maxlen=max(1, cfg.curriculum_promotion_window))

    for episode in range(1, cfg.episodes + 1):
        level_for_episode = map_number
        if cfg.curriculum:
            if cfg.curriculum_adaptive:
                level_for_episode = current_level
            else:
                level_for_episode = min(cfg.curriculum_max_level, 1 + (episode - 1) // cfg.curriculum_step_episodes)
                current_level = level_for_episode
            env.set_map(map_kind=map_kind, map_number=level_for_episode)

        obs = env.reset()
        state_key = _obs_key(env, obs)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = _epsilon_greedy_action(
                q_table=q_table,
                state_key=state_key,
                action_space_n=env.action_space_n,
                epsilon=epsilon,
            )

            next_obs, reward, done, info = env.step(action)
            next_key = _obs_key(env, next_obs)

            if state_key not in q_table:
                q_table[state_key] = np.zeros(env.action_space_n, dtype=np.float32)
            if next_key not in q_table:
                q_table[next_key] = np.zeros(env.action_space_n, dtype=np.float32)

            target = reward
            if not done:
                target += cfg.gamma * float(np.max(q_table[next_key]))

            q_table[state_key][action] += cfg.alpha * (target - q_table[state_key][action])

            state_key = next_key
            total_reward += reward
            steps += 1

            if steps >= cfg.max_steps:
                break

        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

        episode_success = 1.0 if info.get("level_completed", False) else 0.0
        episode_all_collected = 1.0 if info.get("all_collected", False) else 0.0

        reward_history.append(total_reward)
        success_history.append(episode_success)
        all_collected_history.append(episode_all_collected)
        step_history.append(steps)

        if cfg.curriculum and cfg.curriculum_adaptive:
            level_episode_count += 1
            level_success_window.append(episode_success)
            if (
                current_level < cfg.curriculum_max_level
                and level_episode_count >= cfg.curriculum_step_episodes
                and len(level_success_window) == level_success_window.maxlen
            ):
                rolling_success = float(np.mean(level_success_window))
                if rolling_success >= cfg.curriculum_promotion_success:
                    prev_level = current_level
                    current_level += 1
                    level_episode_count = 0
                    level_success_window.clear()
                    epsilon = max(epsilon, cfg.curriculum_level_up_epsilon)
                    print(
                        f"Curriculum promotion: level {prev_level} -> {current_level} | "
                        f"rolling_success={rolling_success:.2%} | epsilon={epsilon:.3f}"
                    )

        if episode % cfg.report_every == 0 or episode == 1:
            avg_reward = float(np.mean(reward_history[-cfg.report_every :]))
            avg_success = float(np.mean(success_history[-cfg.report_every :]))
            avg_all_collected = float(np.mean(all_collected_history[-cfg.report_every :]))
            avg_steps = float(np.mean(step_history[-cfg.report_every :]))
            current_level_for_log = current_level if cfg.curriculum else map_number
            print(
                f"Episode {episode:4d} | "
                f"level={current_level_for_log:2d} | "
                f"avg_reward={avg_reward:8.2f} | "
                f"all_collected_rate={avg_all_collected:5.2%} | "
                f"success_rate={avg_success:5.2%} | "
                f"avg_steps={avg_steps:6.1f} | "
                f"epsilon={epsilon:.3f}"
            )

        if cfg.preview_every > 0 and episode % cfg.preview_every == 0:
            preview_level = current_level if cfg.curriculum else map_number
            _preview_policy(
                q_table=q_table,
                map_kind=map_kind,
                map_number=preview_level,
                observation_mode=observation_mode,
                local_view_size=local_view_size,
                max_steps=cfg.max_steps,
            )

    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.model_path.open("wb") as f:
        pickle.dump(q_table, f)

    env.close()

    print(f"Training complete. Q-table saved to: {cfg.model_path}")
    return q_table


def load_q_table(model_path: Path | None = None) -> Dict[Tuple[int, ...], np.ndarray]:
    default_path = Path(__file__).resolve().parent / "q_table_bobby.pkl"
    raw_path = model_path or default_path

    if raw_path.is_absolute() or raw_path.exists() or model_path is None:
        resolved_path = raw_path
    else:
        script_relative = Path(__file__).resolve().parent / raw_path
        resolved_path = script_relative if script_relative.exists() else raw_path

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Q-table file not found. "
            f"Checked: '{resolved_path}'"
            + (
                f" and '{Path(__file__).resolve().parent / raw_path}'"
                if not raw_path.is_absolute()
                else ""
            )
        )

    with resolved_path.open("rb") as f:
        data = pickle.load(f)
    return data


def play_trained_agent(
    model_path: Path | None = None,
    map_kind: str = "normal",
    map_number: int = 1,
    observation_mode: str = "compact",
    local_view_size: int = 3,
    render: bool = True,
    max_steps: int = 500,
    render_fps: float = 4.0,
    hold_finish_seconds: float = 2.0,
    wait_for_close: bool = False,
) -> Tuple[float, bool, bool, int]:
    q_table = load_q_table(model_path)

    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=map_number,
        observation_mode=observation_mode,
        local_view_size=local_view_size,
        include_inventory=True,
        headless=not render,
        max_steps=max_steps,
    )

    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    success = False
    all_collected = False

    while not done and steps < max_steps:
        state_key = _obs_key(env, obs)
        action = _select_greedy_action(
            q_table=q_table,
            state_key=state_key,
            action_space_n=env.action_space_n,
        )

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if render:
            env.render()
            if render_fps > 0:
                time.sleep(1.0 / render_fps)

        if info.get("level_completed", False):
            success = True
        if info.get("all_collected", False):
            all_collected = True

    # For some maps/policies, collecting all targets is the practical objective even if
    # the finish tile is not reached in the same run.
    if all_collected and not success:
        success = True

    if render:
        # Keep final frame visible long enough to see result.
        if wait_for_close and env._pygame is not None:
            pygame = env._pygame
            print("Window is open. Close it to continue...")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                time.sleep(0.02)
        elif hold_finish_seconds > 0:
            time.sleep(hold_finish_seconds)

    env.close()

    print(
        f"Play finished | total_reward={total_reward:.2f} | "
        f"all_collected={all_collected} | "
        f"success={success} | steps={steps}"
    )

    return total_reward, success, all_collected, steps


def _preview_policy(
    q_table: Dict[Any, np.ndarray],
    map_kind: str,
    map_number: int,
    observation_mode: str,
    local_view_size: int,
    max_steps: int,
) -> None:
    env = BobbyCarrotEnv(
        map_kind=map_kind,
        map_number=map_number,
        observation_mode=observation_mode,
        local_view_size=local_view_size,
        include_inventory=True,
        headless=False,
        max_steps=max_steps,
    )
    obs = env.reset()
    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < max_steps:
        state_key = _obs_key(env, obs)
        action = _select_greedy_action(
            q_table=q_table,
            state_key=state_key,
            action_space_n=env.action_space_n,
        )
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        steps += 1

    env.close()
    print(f"Preview run | reward={total_reward:.2f} | steps={steps}")


def evaluate_q_table(
    episodes: int = 100,
    model_path: Path | None = None,
    map_kind: str = "normal",
    map_number: int = 1,
    observation_mode: str = "compact",
    local_view_size: int = 3,
    max_steps: int = 500,
) -> Dict[str, float]:
    rewards: List[float] = []
    successes: List[float] = []
    all_collected_list: List[float] = []
    steps_list: List[int] = []

    for _ in range(episodes):
        total_reward, success, all_collected, steps = play_trained_agent(
            model_path=model_path,
            map_kind=map_kind,
            map_number=map_number,
            observation_mode=observation_mode,
            local_view_size=local_view_size,
            render=False,
            max_steps=max_steps,
        )
        rewards.append(total_reward)
        successes.append(1.0 if success else 0.0)
        all_collected_list.append(1.0 if all_collected else 0.0)
        steps_list.append(steps)

    metrics = {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "all_collected_rate": float(np.mean(all_collected_list)),
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(steps_list)),
    }

    print("Evaluation summary")
    print(f"episodes={episodes}")
    print(f"mean_reward={metrics['mean_reward']:.2f}")
    print(f"std_reward={metrics['std_reward']:.2f}")
    print(f"all_collected_rate={metrics['all_collected_rate']:.2%}")
    print(f"success_rate={metrics['success_rate']:.2%}")
    print(f"mean_steps={metrics['mean_steps']:.2f}")

    return metrics


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate Bobby Carrot Q-learning agent")
    parser.add_argument("--eval", action="store_true", dest="eval_mode", help="Run evaluation instead of training")
    parser.add_argument("--play", action="store_true", dest="play_mode", help="Play with trained policy (autonomous gameplay)")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training/evaluation episodes")
    parser.add_argument("--play-episodes", type=int, default=1, help="Number of autonomous play episodes in --play mode")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering in --play mode")
    parser.add_argument("--render-fps", type=float, default=4.0, help="Playback speed for --play mode (frames/sec)")
    parser.add_argument("--hold-finish-seconds", type=float, default=2.0, help="How long to keep window visible after episode")
    parser.add_argument("--wait-close", action="store_true", help="Keep window open until manually closed")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--map-kind", type=str, default="normal", choices=["normal", "egg"], help="Map kind")
    parser.add_argument("--map-number", type=int, default=1, help="Map number")
    parser.add_argument(
        "--observation-mode",
        type=str,
        default="compact",
        choices=["compact", "local", "full"],
        help="Observation type",
    )
    parser.add_argument("--local-view-size", type=int, default=3, help="Odd local window size")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "q_table_bobby.pkl"),
        help="Path to q-table file",
    )
    parser.add_argument("--alpha", type=float, default=0.15, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.998, help="Epsilon decay")
    parser.add_argument("--report-every", type=int, default=50, help="Training log interval")
    parser.add_argument("--preview-every", type=int, default=0, help="Render one preview episode every N training episodes")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning across levels")
    parser.add_argument("--no-curriculum", action="store_false", dest="curriculum", help="Disable curriculum learning")
    parser.add_argument("--curriculum-max-level", type=int, default=5, help="Highest level index used by curriculum")
    parser.add_argument("--curriculum-step-episodes", type=int, default=1500, help="Minimum episodes per level before possible promotion")
    parser.add_argument("--curriculum-adaptive", action="store_true", dest="curriculum_adaptive", help="Promote levels only after sustained success")
    parser.add_argument("--curriculum-static", action="store_false", dest="curriculum_adaptive", help="Use fixed episode-based level schedule")
    parser.add_argument("--curriculum-promotion-window", type=int, default=100, help="Rolling episode window size for promotion checks")
    parser.add_argument("--curriculum-promotion-success", type=float, default=0.9, help="Required rolling success rate to promote level")
    parser.add_argument("--curriculum-level-up-epsilon", type=float, default=0.5, help="Minimum epsilon to reset to when promoting")
    parser.add_argument("--debug-env", action="store_true", help="Enable debug prints from environment step info")
    parser.add_argument("--debug-every", type=int, default=100, help="Print debug info every N steps when debug mode is on")
    parser.set_defaults(curriculum=True, curriculum_adaptive=True)
    return parser


def _main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if args.eval_mode:
        evaluate_q_table(
            episodes=args.episodes,
            model_path=model_path,
            map_kind=args.map_kind,
            map_number=args.map_number,
            observation_mode=args.observation_mode,
            local_view_size=args.local_view_size,
            max_steps=args.max_steps,
        )
        return

    if args.play_mode:
        for i in range(1, args.play_episodes + 1):
            print(f"Play episode {i}/{args.play_episodes}")
            play_trained_agent(
                model_path=model_path,
                map_kind=args.map_kind,
                map_number=args.map_number,
                observation_mode=args.observation_mode,
                local_view_size=args.local_view_size,
                render=not args.no_render,
                max_steps=args.max_steps,
                render_fps=args.render_fps,
                hold_finish_seconds=args.hold_finish_seconds,
                wait_for_close=args.wait_close,
            )
        return

    cfg = QLearningConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        max_steps=args.max_steps,
        report_every=args.report_every,
        preview_every=args.preview_every,
        curriculum=args.curriculum,
        curriculum_max_level=args.curriculum_max_level,
        curriculum_step_episodes=args.curriculum_step_episodes,
        curriculum_adaptive=args.curriculum_adaptive,
        curriculum_promotion_window=args.curriculum_promotion_window,
        curriculum_promotion_success=args.curriculum_promotion_success,
        curriculum_level_up_epsilon=args.curriculum_level_up_epsilon,
        debug_env=args.debug_env,
        debug_every=args.debug_every,
        model_path=model_path,
    )

    train_q_learning(
        map_kind=args.map_kind,
        map_number=args.map_number,
        observation_mode=args.observation_mode,
        local_view_size=args.local_view_size,
        config=cfg,
    )


if __name__ == "__main__":
    _main()
