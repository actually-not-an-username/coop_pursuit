import torch

class RolloutBuffer:
    """
    Given the assumptions:
        - one environment
        - N homogeneous predators (default to 3)
        - one shared policy actor
        - one centralized critic
        - one team reward per timestep (according to simple_tag_v3 the rewards duplicate for all agents)
        - one team value V(s_t)
    """

    def __init__(self,
        rollout_length,
        num_agents,
        obs_dim,
        state_dim,
        hidden_dim,
        device,
        recurrent_layers=1,
    ):
        super().__init__()
        self.rollout_length = rollout_length
        self.num_agents = num_agents
        self.device = device

        # Init the data storing tensors
        self.observations = torch.zeros(
            rollout_length + 1, # T+1, last one is actually used
            num_agents,
            obs_dim,
            device=device,
        )

        self.states = torch.zeros(
            rollout_length + 1,
            state_dim,
            device=device,
        )

        self.actions = torch.zeros(
            rollout_length,
            num_agents,
            dtype=torch.long,
            device=device,
        )

        self.old_log_probs = torch.zeros(
            rollout_length,
            num_agents,
            device=device,
        )

        self.rewards = torch.zeros(
            rollout_length,
            device=device,
        )

        self.values = torch.zeros(
            rollout_length + 1,
            device=device,
        )

        self.advantages = torch.zeros(
            rollout_length,
            device=device,
        )

        self.returns = torch.zeros(
            rollout_length,
            device=device,
        )

        self.actor_hidden_states = torch.zeros(
            rollout_length + 1,
            num_agents,
            recurrent_layers,
            hidden_dim,
            device=device,
        )

        self.actor_rnn_masks = torch.ones(
            rollout_length + 1,
            num_agents,
            1,
            device=device,
        )

        self.critic_hidden_states = torch.zeros(
            rollout_length + 1,
            1,
            recurrent_layers,
            hidden_dim,
            device=device,
        )

        self.critic_rnn_masks = torch.ones(
            rollout_length + 1,
            1,
            1,
            device=device,
        )

        self.step = 0

    def insert(self,
               observation,
               state,
               actor_hidden,
               critic_hidden,
               action,
               log_prob,
               value,
               reward,
               next_observation,
               next_state,
               next_actor_hidden,
               next_critic_hidden,
               next_actor_mask,
               next_critic_mask):
        """
        Save one step of the environment in the RolloutBuffer
        """
        t = self.step

        if t >= self.rollout_length:
            raise RuntimeError("Rollout buffer is full.")

        # Current timestep
        self.observations[t].copy_(observation)
        self.states[t].copy_(state)
        self.actor_hidden_states[t].copy_(actor_hidden)
        self.critic_hidden_states[t].copy_(critic_hidden)
        self.actions[t].copy_(action)
        self.old_log_probs[t].copy_(log_prob)

        self.values[t].copy_(value)
        self.rewards[t].copy_(reward)

        # Next timestep
        self.observations[t + 1].copy_(next_observation)
        self.states[t + 1].copy_(next_state)

        self.actor_hidden_states[t + 1].copy_(next_actor_hidden)
        self.actor_rnn_masks[t + 1].copy_(next_actor_mask)

        self.critic_hidden_states[t + 1].copy_(next_critic_hidden)
        self.critic_rnn_masks[t + 1].copy_(next_critic_mask)

        self.step += 1


    def compute_return_and_advantages(self,
                                      next_value,
                                      gamma= 0.99, # Using the same gamma and lambda as the original MAPPO paper
                                      gae_lambda= 0.95):
        """
        Compute Generalized Advantage Estimation (GAE) using the current buffer.
        """
        # use t+1 for final state value
        self.values[self.step].copy_(next_value.squeeze())
        gae = 0

        for t in reversed(range(self.step)):
            mask = self.critic_rnn_masks[t+1].squeeze() # check if the trajectory continue into t+1 => 1 = yes, 0 = no
            delta_t = self.rewards[t] + gamma * self.values[t+1] * mask - self.values[t]
            gae = delta_t + gamma * gae_lambda * mask * gae # GAE calculation

            # gae is composed of A_t and V => R_t = A_t + V(s_t)
            self.advantages[t] = gae
            self.returns[:self.step] = (self.advantages[:self.step] + self.values[:self.step])


    def recurrent_generator(
        self,
        recurrent_chunk_length,
        advantages = None,
    ):
        """
            Split the current rollout into temporally contiguous
            recurrent sequences.

            Example with T=25 and chunk length=10:

                chunk 1: 0...9
                chunk 2: 10...19
                chunk 3: 20...24
            """

        t = self.step

        if t == 0:
            return

        if advantages is None:
            advantages = self.advantages[:t]

        for start in range(0, t, recurrent_chunk_length):
            end = min(start + recurrent_chunk_length, t)

            # Actor data
            observations = self.observations[start:end]
            actions = self.actions[start:end]
            old_log_probs = self.old_log_probs[start:end]
            actor_masks = self.actor_rnn_masks[start:end]

            # IMPORTANT:
            # During training we only provide the GRU the state at the beginning of the sequence.
            actor_initial_hidden = self.actor_hidden_states[start]

            advantages = self.advantages[start:end]

            # Critic data
            states = self.states[start:end]
            returns = self.returns[start:end]
            critic_masks = self.critic_rnn_masks[start:end]

            critic_initial_hidden = self.critic_hidden_states[start]

            yield {
                "observations": observations,
                "actions": actions,
                "old_log_probs": old_log_probs,
                "advantages": advantages,
                "actor_masks": actor_masks,
                "actor_initial_hidden": actor_initial_hidden,

                "states": states,
                "returns": returns,
                "critic_masks": critic_masks,
                "critic_initial_hidden": critic_initial_hidden,
            }