import random

import torch
import os
import numpy as np
from torch import nn
from torch.distributions import Categorical

from policies.Rnn import RNNLayer
from utils import Utils, EnvUtils
from dotenv import load_dotenv

# Configuration
load_dotenv("config.env")

MAX_STEPS=int(os.getenv("MAX_STEPS", np.nan))
TEST_OBSERVATION_FAILURE_STEPS=int(os.getenv("TEST_OBSERVATION_FAILURE_STEPS", np.nan))

class Actor(nn.Module):
    """
        Shared decentralized actor.

        Each agent receives:
            - its own local observation
            - its own GRU hidden state

        It produces:
            - an action distribution
            - the next GRU hidden state

        The network parameters can be shared between al homogeneous adversaries,
        but each adversary MUST maintain its own recurrent hidden state.
        """

    def __init__(self, observation_dim, hidden_dim = 64, action_dim = 5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim

        # MLP with orthogonal init as implemented in MAPPO
        self.mlp = Utils.create_mlp_layers(observation_dim, hidden_dim)
        self.rnn = RNNLayer(hidden_dim, hidden_dim)

        # keep the last gain small so the initial distribution is close to uniform, in theory given
        # n epochs this clipping shouldn't affect the final output
        self.action_head = Utils.init_orthogonal_layer(nn.Linear(hidden_dim, action_dim), gain=0.01)

    def forward(self, observation, hidden_state, mask):
        x = self.mlp(observation)
        x, next_hidden_state = self.rnn(x, hidden_state, mask)
        logits = self.action_head(x)
        dist = Categorical(logits=logits)

        return dist, next_hidden_state

    @torch.no_grad()
    def act(self, observation, hidden_state, mask):
        dist, next_hidden_state = self.forward(observation, hidden_state, mask)
        action = dist.sample() # This action selection allows exploration
        # action = dist.probs.argmax(dim=-1) # This action allows exploitation
        log_prob = dist.log_prob(action)
        return action, log_prob, next_hidden_state

    def evaluate_actions(self, observation, hidden_state, actions, mask):
        dist, next_hidden_state = self.forward(observation, hidden_state, mask)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, next_hidden_state

    @torch.no_grad()
    def evaluate_policy(self,
                device: torch.device,
                seed: int,
                episodes: int,
                action_mode: str = "argmax",
                masking : bool = False, #adding to make training not masked, eval masked
        ) -> list[dict[str, float | bool | int | None]]:
            """Evaluate an actor or random policy on a fixed set of episode seeds."""
            if action_mode not in {"argmax", "sample", "random"}:
                raise ValueError(f"Unknown evaluation action mode: {action_mode}")

            self.eval()
            env = EnvUtils.get_env()
            episode_returns = []

            try:
                predators = [agent for agent in env.possible_agents if "adversary" in agent]
                prey_agents = [agent for agent in env.possible_agents if "adversary" not in agent]
                for episode_index in range(episodes):
                    episode_seed = seed + episode_index
                    # Make sampled-policy evaluation repeatable without changing training.
                    observations, _ = env.reset(seed=episode_seed)
                    for agent_index, agent in enumerate(env.possible_agents):
                        env.action_space(agent).seed(episode_seed + agent_index + 1)

                    hidden_state = Utils.init_hidden_state(len(predators), self.hidden_dim)
                    mask = Utils.init_rnn_mask(len(predators), 1)
                    episode_return = 0.0
                    capture_step = None

                    # As a design choice we want only one attacker to be able to see the rest of them will go blind :)
                    # and also for a certain number of steps
                    starting_sensor_loss_step = random.randint(0, MAX_STEPS-TEST_OBSERVATION_FAILURE_STEPS)
                    the_chosen = predators[random.randint(0, len(predators) - 1)]

                    for step_index in range(MAX_STEPS):
                        if not env.agents:
                            break
                        #masking now checked
                        if starting_sensor_loss_step <= step_index < (starting_sensor_loss_step + TEST_OBSERVATION_FAILURE_STEPS) and masking:
                            for agent in predators:
                                if agent != the_chosen:
                                    observations[agent] = np.zeros_like(observations[agent])

                        predator_obs = torch.as_tensor(
                            np.stack([observations[agent] for agent in predators]),
                            dtype=torch.float32,
                            device=device,
                        )

                        actions = {
                            agent: env.action_space(agent).sample()
                            for agent in env.agents
                            if agent not in predators
                        }
                        if action_mode == "random":
                            actions.update({
                                agent: env.action_space(agent).sample()
                                for agent in predators
                            })
                        else:
                            dist, next_hidden_state = self.forward(predator_obs, hidden_state, mask)
                            if action_mode == "argmax":
                                predator_actions = dist.probs.argmax(dim=-1)
                            else:
                                predator_actions = dist.sample()
                            for index, agent in enumerate(predators):
                                actions[agent] = int(predator_actions[index].item())
                            hidden_state = next_hidden_state

                        observations, rewards, terminations, truncations, _infos = env.step(actions)
                        episode_return += float(np.mean([rewards[agent] for agent in predators]))
                        if capture_step is None and all(
                                terminations.get(agent, False) for agent in prey_agents
                        ):
                            capture_step = step_index + 1
                        episode_done = all(
                            terminations.get(agent, False) or truncations.get(agent, False)
                            for agent in predators
                        )
                        mask = torch.full(
                            (len(predators), 1), 0.0 if episode_done else 1.0,
                            dtype=torch.float32,
                            device=device,
                        )
                        if episode_done:
                            break

                    episode_returns.append({
                        "return": episode_return,
                        "captured": capture_step is not None,
                        "capture_step": capture_step,
                    })
            finally:
                env.close()

            return episode_returns
