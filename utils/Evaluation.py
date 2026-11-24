"""Evaluation episodes (validation during training and the Variation 3 test sweep)."""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import torch

from utils import Utils
from utils.Blackout import BlackoutApplier, BlackoutSpec
from utils.Config import CONFIG
from utils.EnvUtils import seed_action_spaces, split_agents
from utils.Prey import prey_actions, prey_rng


def max_angular_gap(prey_pos: np.ndarray, pursuer_positions: np.ndarray) -> float:
    """Largest angular gap (degrees) between pursuers as seen from the prey. 360 = all on one bearing."""
    offsets = pursuer_positions - prey_pos
    angles = np.sort(np.arctan2(offsets[:, 1], offsets[:, 0]))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
    return float(np.degrees(gaps.max()))


class EpisodeMetrics:
    """Behaviour metrics from world positions after every step."""

    def __init__(self, world):
        self.world = world
        self.pursuers = [a for a in world.agents if a.adversary]
        self.prey = [a for a in world.agents if not a.adversary][0]
        self.pairs = list(combinations(range(len(self.pursuers)), 2))
        self.in_contact = {pair: False for pair in self.pairs}
        self.distances, self.spreads, self.gaps = [], [], []
        self.collisions = 0

    def record(self) -> None:
        positions = np.array([a.state.p_pos for a in self.pursuers])
        prey_pos = self.prey.state.p_pos
        self.distances.append(float(np.linalg.norm(positions - prey_pos, axis=1).mean()))
        pair_distances = []
        for i, j in self.pairs:
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            pair_distances.append(distance)
            touching = distance < self.pursuers[i].size + self.pursuers[j].size
            if touching and not self.in_contact[(i, j)]:
                self.collisions += 1  # count contact events
            self.in_contact[(i, j)] = touching
        self.spreads.append(float(np.mean(pair_distances)))
        self.gaps.append(max_angular_gap(prey_pos, positions))

    def summary(self) -> dict:
        return {
            "mean_distance": float(np.mean(self.distances)) if self.distances else float("nan"),
            "spread": float(np.mean(self.spreads)) if self.spreads else float("nan"),
            "max_angular_gap": float(np.mean(self.gaps)) if self.gaps else float("nan"),
            "collisions": self.collisions,
        }


@torch.no_grad()
def run_eval_episode(env, actor, device: torch.device, episode_seed: int,
                     blackout: BlackoutSpec | None, policy: str = "argmax") -> dict:
    """One evaluation episode. policy: 'argmax', 'sample' or 'random' (actor ignored)."""
    if policy not in {"argmax", "sample", "random"}:
        raise ValueError(f"Unknown policy mode: {policy}")
    predators, prey_agents = split_agents(env)
    observations, _ = env.reset(seed=episode_seed)
    seed_action_spaces(env, episode_seed)
    rng = prey_rng(episode_seed)
    applier = BlackoutApplier(blackout)
    metrics = EpisodeMetrics(env.unwrapped.world)

    hidden_dim = actor.hidden_dim if actor is not None else 1
    hidden_state = Utils.init_hidden_state(len(predators), hidden_dim)
    mask = Utils.init_rnn_mask(len(predators), 1)
    env_return = 0.0
    capture_step = None
    steps = 0

    for step_index in range(CONFIG.max_steps):
        if not env.agents:
            break
        raw_obs = np.stack([observations[agent] for agent in predators])
        seen_obs = applier.actor_inputs(raw_obs, step_index)

        actions = prey_actions(env, prey_agents, rng)
        if policy == "random":
            actions.update({agent: env.action_space(agent).sample() for agent in predators})
        else:
            obs_tensor = torch.as_tensor(seen_obs, dtype=torch.float32, device=device)
            dist, hidden_state = actor.forward(obs_tensor, hidden_state, mask)
            chosen = dist.probs.argmax(dim=-1) if policy == "argmax" else dist.sample()
            for index, agent in enumerate(predators):
                actions[agent] = int(chosen[index].item())

        observations, rewards, terminations, truncations, _ = env.step(actions)
        steps += 1
        metrics.record()
        env_return += float(np.mean([rewards[agent] for agent in predators]))
        if capture_step is None and all(terminations.get(agent, False) for agent in prey_agents):
            capture_step = step_index + 1
        done = all(terminations.get(a, False) or truncations.get(a, False) for a in predators)
        mask = torch.full((len(predators), 1), 0.0 if done else 1.0, dtype=torch.float32, device=device)
        if done:
            break

    captured = capture_step is not None
    return {
        "captured": captured,
        "capture_step": capture_step,
        # capture_step <= start means the capture happened before any blinded step was played
        "captured_before_blackout": (captured and capture_step <= blackout.start) if blackout else None,
        "env_return": env_return,
        "steps": steps,
        **metrics.summary(),
    }


def evaluate(env, actor, device: torch.device, first_seed: int, episodes: int,
             blackout_for_seed: Callable[[int], BlackoutSpec | None], policy: str = "argmax") -> dict:
    """Aggregate capture rate and env return over a fixed block of seeds."""
    if actor is not None:
        actor.eval()
    results = [
        run_eval_episode(env, actor, device, seed, blackout_for_seed(seed), policy)
        for seed in range(first_seed, first_seed + episodes)
    ]
    if actor is not None:
        actor.train()
    return {
        "capture_rate": float(np.mean([r["captured"] for r in results])),
        "env_return": float(np.mean([r["env_return"] for r in results])),
    }
