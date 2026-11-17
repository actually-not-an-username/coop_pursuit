from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mpe2 import simple_tag_v3


class VectorTagEnv(gym.Env):
    """Gymnasium adapter for one cooperative-pursuit environment.

    The vector action controls all predators. The prey remains random, and the
    observation contains both decentralized predator observations and the
    centralized state used by the critic.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int,
        shaping: bool,
        terminate_on_success: bool = True,
    ) -> None:
        super().__init__()
        self.env = simple_tag_v3.parallel_env(
            max_cycles=max_steps,
            terminate_on_success=terminate_on_success,
        )
        self.shaping = shaping
        self.predators = tuple(
            agent
            for agent in self.env.possible_agents
            if "adversary" in agent
        )
        self.prey = tuple(
            agent
            for agent in self.env.possible_agents
            if "adversary" not in agent
        )

        predator_space = self.env.observation_space(self.predators[0])
        predator_count = len(self.predators)
        self.observation_space = spaces.Dict({
            "predator_observations": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(predator_count, predator_space.shape[0]),
                dtype=np.float32,
            ),
            "state": self.env.state_space,
        })
        self.action_space = spaces.MultiDiscrete([
            self.env.action_space(agent).n
            for agent in self.predators
        ])
        self.episode_step = 0

    def _observation(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "predator_observations": np.stack([
                observations[agent]
                for agent in self.predators
            ]).astype(np.float32),
            "state": self.env.state().astype(np.float32),
        }

    def _shaping_reward(self) -> float:
        world = self.env.unwrapped.world
        targets = [agent for agent in world.agents if not agent.adversary]
        adversaries = [agent for agent in world.agents if agent.adversary]
        return -0.1 * sum(
            min(
                np.linalg.norm(target.state.p_pos - adversary.state.p_pos)
                for target in targets
            )
            for adversary in adversaries
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        observations, _ = self.env.reset(seed=seed, options=options)
        if seed is not None:
            for index, agent in enumerate(self.env.possible_agents):
                self.env.action_space(agent).seed(seed + index + 1)
        self.episode_step = 0
        return self._observation(observations), {}

    def step(
        self,
        predator_actions: np.ndarray,
    ) -> tuple[
        dict[str, np.ndarray],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        actions = {
            agent: self.env.action_space(agent).sample()
            for agent in self.prey
            if agent in self.env.agents
        }
        actions.update({
            agent: int(predator_actions[index])
            for index, agent in enumerate(self.predators)
            if agent in self.env.agents
        })

        observations, rewards, terminations, truncations, _ = self.env.step(actions)
        self.episode_step += 1

        raw_team_reward = float(np.mean([
            rewards[agent]
            for agent in self.predators
        ]))
        shaping_reward = self._shaping_reward() if self.shaping else 0.0
        reward = raw_team_reward + shaping_reward
        terminated = all(terminations.get(agent, False) for agent in self.predators)
        truncated = all(truncations.get(agent, False) for agent in self.predators)

        if terminated or truncated:
            next_observation = {
                "predator_observations": np.zeros(
                    self.observation_space["predator_observations"].shape,
                    dtype=np.float32,
                ),
                "state": np.zeros(
                    self.observation_space["state"].shape,
                    dtype=np.float32,
                ),
            }
        else:
            next_observation = self._observation(observations)

        return next_observation, reward, terminated, truncated, {
            "captured": terminated,
            "episode_step": self.episode_step,
            "raw_team_reward": raw_team_reward,
            "shaping_reward": shaping_reward,
        }

    def close(self) -> None:
        self.env.close()

