"""
Run configuration is read.
Values can be overridden by environment variables
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields

from dotenv import load_dotenv

load_dotenv("config.env")


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)).replace("_", ""))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() == "true"


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip().strip('"')


@dataclass
class Config:
    # Training
    seed: int = _env_int("GLOBAL_SEED", 1)
    num_episodes: int = _env_int("NUM_EPISODES", 2048)
    rollout_episodes: int = _env_int("ROLLOUT_EPISODES", 32)
    max_steps: int = _env_int("MAX_STEPS", 100)
    hidden_dim: int = _env_int("HIDDEN_DIM", 128)
    shaping: bool = _env_bool("SHAPING", True)
    render: bool = _env_bool("RENDER", False)

    # Environment / scenario
    num_pursuers: int = _env_int("NUM_PURSUERS", 3)
    prey_type: str = _env_str("PREY_TYPE", "random") 
    flee_prob: float = _env_float("FLEE_PROB", 0.5)
    flee_boundary: float = _env_float("FLEE_BOUNDARY", 0.8)
    flee_push: float = _env_float("FLEE_PUSH", 2.5)

    # Masked training
    masked_training: bool = _env_bool("MASKED_TRAINING", False)
    train_blackout_prob: float = _env_float("TRAIN_BLACKOUT_PROB", 0.8)
    train_blackout_start_max: int = _env_int("TRAIN_BLACKOUT_START_MAX", 30)
    train_blackout_min_len: int = _env_int("TRAIN_BLACKOUT_MIN_LEN", 5)
    train_blackout_max_len: int = _env_int("TRAIN_BLACKOUT_MAX_LEN", 40)
    train_blackout_permanent_prob: float = _env_float("TRAIN_BLACKOUT_PERMANENT_PROB", 0.2)
    train_blackout_start_min: int = _env_int("TRAIN_BLACKOUT_START_MIN", 1)

    
    keep_proprioception: bool = _env_bool("KEEP_PROPRIOCEPTION", False)
    blind_flag: bool = _env_bool("BLIND_FLAG", False)                    
    critic_blind_mask: bool = _env_bool("CRITIC_BLIND_MASK", False)

    # Train-until-competent safety net: after num_episodes, keep training in chunks of
    # extension_episodes while best no-blackout validation capture < val_threshold,
    # up to max_episodes. val_threshold <= 0 disables it.
    val_threshold: float = _env_float("VAL_THRESHOLD", 0.0)
    extension_episodes: int = _env_int("EXTENSION_EPISODES", 320)
    max_episodes: int = _env_int("MAX_EPISODES", 0)

    # Validation during training
    eval_interval_updates: int = _env_int("EVAL_INTERVAL_UPDATES", 10)
    eval_episodes: int = _env_int("EVAL_EPISODES", 100)
    eval_seed: int = _env_int("EVAL_SEED", 10_000)
    val_blackout_length: int = _env_int("VAL_BLACKOUT_LENGTH", 20)

    # Blackout timing shared by validation and test (start drawn uniformly in [min, max])
    blackout_start_min: int = _env_int("BLACKOUT_START_MIN", 1)
    blackout_start_max: int = _env_int("BLACKOUT_START_MAX", 30)

    # Test sweep
    test_episodes: int = _env_int("TEST_EPISODES", 300)
    test_seed: int = _env_int("TEST_SEED", 20_000)

    # Output
    runs_dir: str = _env_str("RUNS_DIR", "runs")

    def update(self, **overrides) -> None:
        valid = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in valid:
                raise KeyError(f"Unknown config field: {key}")
            setattr(self, key, value)
        self.validate()

    def validate(self) -> None:
        if self.prey_type not in {"random", "fleeing"}:
            raise ValueError(f"PREY_TYPE must be 'random' or 'fleeing', got {self.prey_type!r}")
        if self.num_pursuers < 2:
            raise ValueError("Need at least 2 pursuers so that N-1 >= 1 can be blinded.")
        longest_finite = max(40, self.val_blackout_length)
        if self.blackout_start_max + longest_finite > self.max_steps:
            raise ValueError("blackout_start_max + 40 must fit inside max_steps.")
        if self.blackout_start_min < 1 or self.train_blackout_start_min < 1:
            raise ValueError("blackout start must be >= 1 so 'hold' has an observation to hold.")
        for name in ("num_episodes", "extension_episodes", "max_episodes"):
            value = getattr(self, name)
            if value and value % self.rollout_episodes:
                raise ValueError(f"{name}={value} must be a multiple of rollout_episodes="
                                 f"{self.rollout_episodes} (avoids a truncated final update).")

    def as_dict(self) -> dict:
        return asdict(self)


CONFIG = Config()
CONFIG.validate()
