from __future__ import annotations

import torch


class VectorRolloutBuffer:
    """Fixed-length recurrent rollout for multiple parallel environments."""

    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        hidden_dim: int,
        device: torch.device,
        recurrent_layers: int = 1,
    ) -> None:
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.step = 0

        self.observations = torch.zeros(
            rollout_length + 1, num_envs, num_agents, obs_dim, device=device
        )
        self.states = torch.zeros(
            rollout_length + 1, num_envs, state_dim, device=device
        )
        self.actions = torch.zeros(
            rollout_length, num_envs, num_agents, dtype=torch.long, device=device
        )
        self.old_log_probs = torch.zeros(
            rollout_length, num_envs, num_agents, device=device
        )
        self.rewards = torch.zeros(rollout_length, num_envs, device=device)
        self.values = torch.zeros(rollout_length + 1, num_envs, device=device)
        self.advantages = torch.zeros(rollout_length, num_envs, device=device)
        self.returns = torch.zeros(rollout_length, num_envs, device=device)

        self.actor_hidden_states = torch.zeros(
            rollout_length + 1,
            num_envs,
            num_agents,
            recurrent_layers,
            hidden_dim,
            device=device,
        )
        self.actor_rnn_masks = torch.ones(
            rollout_length + 1, num_envs, num_agents, 1, device=device
        )
        self.critic_hidden_states = torch.zeros(
            rollout_length + 1,
            num_envs,
            recurrent_layers,
            hidden_dim,
            device=device,
        )
        self.critic_rnn_masks = torch.ones(
            rollout_length + 1, num_envs, 1, device=device
        )

    def insert(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        actor_hidden: torch.Tensor,
        critic_hidden: torch.Tensor,
        actor_mask: torch.Tensor,
        critic_mask: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        next_state: torch.Tensor,
        next_actor_hidden: torch.Tensor,
        next_critic_hidden: torch.Tensor,
        next_actor_mask: torch.Tensor,
        next_critic_mask: torch.Tensor,
    ) -> None:
        t = self.step
        if t >= self.rollout_length:
            raise RuntimeError("Rollout buffer is full.")

        self.observations[t].copy_(observation)
        self.states[t].copy_(state)
        self.actor_hidden_states[t].copy_(actor_hidden)
        self.critic_hidden_states[t].copy_(critic_hidden)
        self.actor_rnn_masks[t].copy_(actor_mask)
        self.critic_rnn_masks[t].copy_(critic_mask)
        self.actions[t].copy_(action)
        self.old_log_probs[t].copy_(log_prob)
        self.values[t].copy_(value)
        self.rewards[t].copy_(reward)

        self.observations[t + 1].copy_(next_observation)
        self.states[t + 1].copy_(next_state)
        self.actor_hidden_states[t + 1].copy_(next_actor_hidden)
        self.critic_hidden_states[t + 1].copy_(next_critic_hidden)
        self.actor_rnn_masks[t + 1].copy_(next_actor_mask)
        self.critic_rnn_masks[t + 1].copy_(next_critic_mask)
        self.step += 1

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.values[self.step].copy_(next_value.reshape(self.num_envs))
        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.step)):
            mask = self.critic_rnn_masks[t + 1].squeeze(-1)
            delta = (
                self.rewards[t]
                + gamma * self.values[t + 1] * mask
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * mask * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def recurrent_generator(
        self,
        recurrent_chunk_length: int,
        advantages: torch.Tensor | None = None,
    ):
        """Yield time chunks batched across every environment and agent."""
        if advantages is None:
            advantages = self.advantages[:self.step]

        for start in range(0, self.step, recurrent_chunk_length):
            end = min(start + recurrent_chunk_length, self.step)
            length = end - start
            actor_batch = self.num_envs * self.num_agents

            actor_advantages = (
                advantages[start:end]
                .unsqueeze(-1)
                .expand(length, self.num_envs, self.num_agents)
                .reshape(length, actor_batch)
            )
            yield {
                "observations": self.observations[start:end].reshape(
                    length, actor_batch, -1
                ),
                "actions": self.actions[start:end].reshape(length, actor_batch),
                "old_log_probs": self.old_log_probs[start:end].reshape(
                    length, actor_batch
                ),
                "advantages": actor_advantages,
                "actor_masks": self.actor_rnn_masks[start:end].reshape(
                    length, actor_batch, 1
                ),
                "actor_initial_hidden": self.actor_hidden_states[start].reshape(
                    actor_batch,
                    self.actor_hidden_states.shape[-2],
                    self.actor_hidden_states.shape[-1],
                ),
                "states": self.states[start:end],
                "returns": self.returns[start:end],
                "critic_masks": self.critic_rnn_masks[start:end],
                "critic_initial_hidden": self.critic_hidden_states[start],
            }

