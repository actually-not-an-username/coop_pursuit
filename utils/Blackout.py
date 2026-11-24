"""
Some pursuers lose their observation for a window of steps.

Only pursuers observations are affected. The critic always uses env.state(), and is never blinded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.Config import CONFIG

FILL_METHODS = ("zero", "hold")


@dataclass(frozen=True)
class BlackoutSpec:
    start: int
    length: int | None          # None = permanent
    blinded: tuple[int, ...]    # pursuer indices
    fill: str = "zero"

    def active(self, step: int) -> bool:
        if step < self.start:
            return False
        return self.length is None or step < self.start + self.length

    @property
    def length_label(self) -> str:
        return "permanent" if self.length is None else str(self.length)


# simple_tag pursuer observation layout: [own vel (2), own pos (2), landmarks, other agents, prey vel].
# With keep_proprioception, a blackout removes external sensing only; the pursuer still knows
# its own velocity and position (entries 0-3), like a robot whose camera fails but odometry works.
PROPRIO_DIMS = 4


class BlackoutApplier:
    """Applies a BlackoutSpec step by step and remembers each pursuer's last real observation."""

    def __init__(self, spec: BlackoutSpec | None):
        if spec is not None and spec.fill not in FILL_METHODS:
            raise ValueError(f"Unknown fill method {spec.fill!r}")
        self.spec = spec
        self.last_seen: np.ndarray | None = None

    def blind_mask(self, step: int, num_pursuers: int) -> np.ndarray:
        """1.0 for pursuers blind at this step, else 0.0. Shape [num_pursuers]."""
        mask = np.zeros(num_pursuers, dtype=np.float32)
        if self.spec is not None and self.spec.active(step):
            mask[list(self.spec.blinded)] = 1.0
        return mask

    def apply(self, observations: np.ndarray, step: int) -> np.ndarray:
        """observations: [num_pursuers, obs_dim]. Returns what the actors actually see."""
        if self.last_seen is None:
            self.last_seen = observations.copy()
        if self.spec is None or not self.spec.active(step):
            self.last_seen = observations.copy()
            return observations

        start = PROPRIO_DIMS if CONFIG.keep_proprioception else 0
        seen = observations.copy()
        for index in range(len(seen)):
            if index in self.spec.blinded:
                seen[index, start:] = 0.0 if self.spec.fill == "zero" else self.last_seen[index, start:]
            else:
                self.last_seen[index] = observations[index]
        return seen

    def actor_inputs(self, observations: np.ndarray, step: int) -> np.ndarray:
        """What the actor network receives: the (blinded) observation, plus a blind flag column."""
        seen = self.apply(observations, step)
        if not CONFIG.blind_flag:
            return seen
        flag = self.blind_mask(step, len(observations))[:, None]
        return np.concatenate([seen, flag], axis=1).astype(np.float32)


def actor_obs_dim(env_obs_dim: int) -> int:
    return env_obs_dim + (1 if CONFIG.blind_flag else 0)


def critic_state_dim(env_state_dim: int, num_pursuers: int) -> int:
    return env_state_dim + (num_pursuers if CONFIG.critic_blind_mask else 0)


@dataclass(frozen=True)
class PairedDraw:
    """Per-seed randomness shared by every method and condition on that seed."""
    start: int
    blind_order: tuple[int, ...]

    def spec(self, length: int | None, num_blinded: int, fill: str) -> BlackoutSpec:
        return BlackoutSpec(self.start, length, tuple(sorted(self.blind_order[:num_blinded])), fill)


def paired_draw(episode_seed: int, num_pursuers: int) -> PairedDraw:
    """Blackout start step and blinding order fixed by the episode seed alone."""
    rng = np.random.default_rng([episode_seed, 11])
    start = int(rng.integers(CONFIG.blackout_start_min, CONFIG.blackout_start_max + 1))
    order = tuple(int(i) for i in rng.permutation(num_pursuers))
    return PairedDraw(start, order)


def validation_blackout(episode_seed: int, num_pursuers: int) -> BlackoutSpec:
    """Fixed hard blackout used during training-time validation: N-1 blinded, zero fill."""
    return paired_draw(episode_seed, num_pursuers).spec(
        CONFIG.val_blackout_length, num_pursuers - 1, "zero"
    )


def random_training_blackout(episode_seed: int, num_pursuers: int) -> BlackoutSpec | None:
    """Masked-MAPPO training: random start, length and blinded set for every episode."""
    rng = np.random.default_rng([episode_seed, 13])
    if rng.random() >= CONFIG.train_blackout_prob:
        return None
    start = int(rng.integers(CONFIG.train_blackout_start_min, CONFIG.train_blackout_start_max + 1))
    if rng.random() < CONFIG.train_blackout_permanent_prob:
        length = None
    else:
        length = int(rng.integers(CONFIG.train_blackout_min_len, CONFIG.train_blackout_max_len + 1))
    num_blinded = int(rng.integers(1, num_pursuers))  # 1 .. N-1
    blinded = tuple(sorted(int(i) for i in rng.choice(num_pursuers, num_blinded, replace=False)))
    return BlackoutSpec(start, length, blinded, "zero")
