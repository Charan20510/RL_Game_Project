"""Modular RL training system for Bobby Carrot.

Implements PPO, Rainbow DQN, and ICM (Intrinsic Curiosity Module)
with shared infrastructure for neural networks, replay buffers,
and training orchestration.
"""

from .config import PPOConfig, RainbowConfig, ICMConfig, TrainingConfig, LevelConfig
from .networks import CNNEncoder, ObservationPreprocessor
from .ppo import PPOAgent
from .rainbow import RainbowAgent
from .icm import ICMModule
from .mcts_eval import MCTSEvaluator

__all__ = [
    "PPOConfig",
    "RainbowConfig",
    "ICMConfig",
    "TrainingConfig",
    "LevelConfig",
    "CNNEncoder",
    "ObservationPreprocessor",
    "PPOAgent",
    "RainbowAgent",
    "ICMModule",
    "MCTSEvaluator",
]
