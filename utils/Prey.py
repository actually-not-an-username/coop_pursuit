"""Prey policies.

random  : uniform random discrete action (seeded through the env action space).
fleeing : with probability ``flee_prob`` move away from the nearest pursuer, there is a push
          back toward the centre once the prey is beyond ``flee_boundary`` on an axis;
          otherwise take a uniformly random action. The soft push keeps the faster prey
          inside a bounded area without walls, so it can only be caught by cornering.
"""
from __future__ import annotations

import numpy as np

from utils.Config import CONFIG

# Discrete MPE actions: 0 no-op, 1 -x, 2 +x, 3 -y, 4 +y
_MOVE_DIRECTIONS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])


def prey_rng(episode_seed: int) -> np.random.Generator:
    """Per-episode RNG for the prey's flee/random coin flips (independent of blackout RNG)."""
    return np.random.default_rng([episode_seed, 7])


def flee_direction(prey_pos: np.ndarray, pursuer_positions: np.ndarray,
                   boundary: float, push: float) -> np.ndarray:
    offsets = prey_pos - pursuer_positions
    nearest = offsets[np.argmin(np.linalg.norm(offsets, axis=1))]
    norm = np.linalg.norm(nearest)
    away = nearest / norm if norm > 1e-8 else np.zeros(2)
    excess = prey_pos - np.clip(prey_pos, -boundary, boundary)
    return away - push * excess


def flee_action(direction: np.ndarray) -> int:
    if np.linalg.norm(direction) < 1e-8:
        return 0
    return int(np.argmax(_MOVE_DIRECTIONS @ direction)) + 1


def prey_actions(env, prey_agents: list[str], rng: np.random.Generator) -> dict[str, int]:
    """Actions for every live prey agent under the configured prey type."""
    live = [agent for agent in prey_agents if agent in env.agents]
    if CONFIG.prey_type == "random":
        return {agent: env.action_space(agent).sample() for agent in live}

    world = env.unwrapped.world
    pursuer_positions = np.array([a.state.p_pos for a in world.agents if a.adversary])
    prey_bodies = [a for a in world.agents if not a.adversary]
    actions = {}
    for agent, body in zip(prey_agents, prey_bodies):
        if agent not in live:
            continue
        if rng.random() < CONFIG.flee_prob:
            direction = flee_direction(body.state.p_pos, pursuer_positions,
                                       CONFIG.flee_boundary, CONFIG.flee_push)
            actions[agent] = flee_action(direction)
        else:
            actions[agent] = int(rng.integers(5))
    return actions
