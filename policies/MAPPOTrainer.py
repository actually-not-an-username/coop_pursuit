import torch

from utils.RolloutBuffer import RolloutBuffer


class MAPPOTrainer:
    def __init__(self,
                 actor,
                 actor_optimizer,
                 critic,
                 critic_optimizer,
                 clip_param = 0.2,
                 entropy_coef = 0.01,
                 value_loss_coef = 1.0,
                 max_grad_norm = 10.0,
                 ppo_epochs = 10,
                 recurrent_chunk_length = 10,
                 huber_delta = 10.0):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.recurrent_chunk_length = recurrent_chunk_length
        self.huber_delta = huber_delta

    def update_actor(self, buffer: RolloutBuffer):
        t = buffer.step

        # Process advantages in a separate structure to avoid messing with the originals
        normalized_advantages = buffer.advantages[:t].clone()
        normalized_advantages = (normalized_advantages - normalized_advantages.mean()) / (normalized_advantages.std(unbiased=False) + 1e-5)

        policy_losses = []
        entropy_records = []
        ratios = []
        clip_fractions = []
        chunk_lengths = []

        for chunk in buffer.recurrent_generator(self.recurrent_chunk_length, normalized_advantages):
            obs = chunk["observations"]
            actions = chunk["actions"]
            old_log_probs = chunk["old_log_probs"]
            mask = chunk["actor_masks"]
            initial_hidden_state = chunk["actor_initial_hidden"]
            advantages = chunk["advantages"]
            chunk_lengths.append(obs.shape[0])

            # This code modifies the shape of the advantages from [recurrent_chunk_lenght] to [recurrent_chunk_lenght, number of agents]
            # interestingly enough expand does not copies the data but rather broadcast a view of the data hence modifications
            # to this tensor s.b avoided
            agent_advantages = advantages.unsqueeze(-1) # Add a new dimension to the tensor
            agent_advantages = agent_advantages.expand(-1, buffer.num_agents) # Now make each agent has their own advantage

            # reevaluate the actions from the chunk (old policy)
            new_log_probs, entropy, _ = self.actor.evaluate_actions(obs, initial_hidden_state, actions, mask)
            # PPO probability ratio -> pi_new(a|s)/pi_old(a|s)
            ratio = torch.exp(new_log_probs - old_log_probs)
            ratios.append(ratio.mean())

            # surrogate objective 1 = unclipped ratio (per agent)
            surrogate_obj_1 = ratio * agent_advantages
            # surrogate objective 2 = clipped ratio
            surrogate_obj_2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * agent_advantages

            policy_loss = -torch.min(surrogate_obj_1, surrogate_obj_2).mean()
            policy_losses.append(policy_loss)

            mean_entropy = entropy.mean()
            entropy_records.append(mean_entropy)

            # calculate how many samples needed clipping
            clip_factor = (torch.abs(ratio - 1.0) > self.clip_param).float().mean()
            clip_fractions.append(clip_factor)

        # Since each chunk can have different sizes we should do a weighted mean rather than just a regular mean
        policy_loss_weights = torch.tensor(chunk_lengths, dtype=torch.float32, device=policy_losses[0].device)
        weight = policy_loss_weights / policy_loss_weights.sum()

        policy_loss = (torch.stack(policy_losses) * weight).sum()
        entropy = (torch.stack(entropy_records) * weight).sum()
        actor_loss = (policy_loss - self.entropy_coef * entropy)

        # gradient descent
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        mean_ratio = torch.stack(ratios).mean()
        mean_clip_fraction = torch.stack(clip_fractions).mean()

        return {
            "actor_loss": actor_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "ratio": mean_ratio.item(),
            "clip_fraction": mean_clip_fraction.item(),
        }

    def update_critic(self, buffer: RolloutBuffer):
        t = buffer.step
        value_losses = []
        chunk_lengths = []

        for chunk in buffer.recurrent_generator(self.recurrent_chunk_length):
            states = chunk["states"]
            returns = chunk["returns"]
            masks = chunk["critic_masks"]
            initial_hidden = chunk["critic_initial_hidden"]
            chunk_lengths.append(states.shape[0])

            # states has the shape [recurrent_chunk_length, batch_size, state_dim]
            # and the requirement is [recurrent_chunk_length, state_dim]
            states = states.unsqueeze(1)

            predicted_value, _ = self.critic(states, initial_hidden, masks)
            predicted_value = predicted_value.squeeze(-1)

            if predicted_value.dim() == 2:
                predicted_value = predicted_value.squeeze(-1)

            # Calculate the loss -> Instead of MSE the authors of MAPPO use Hubber Loss
            value_loss = torch.nn.functional.huber_loss(predicted_value, returns, delta=self.huber_delta)
            value_losses.append(value_loss)

        # Since each chunk can have different sizes we should do a weighted mean rather than just a regular mean
        value_loss_weights = torch.tensor(chunk_lengths, dtype=torch.float32, device=value_losses[0].device)
        weight = value_loss_weights / value_loss_weights.sum()

        value_loss = (torch.stack(value_losses) * weight).sum()
        critic_loss = (self.value_loss_coef * value_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "value_loss": value_loss.item(),
        }

    def update(self, buffer: RolloutBuffer):
        final_actor_result = {}
        final_critic_result = {}

        for epoch in range(self.ppo_epochs):
            final_actor_result = self.update_actor(buffer)
            final_critic_result = self.update_critic(buffer)

            print(
                f"  PPO Epoch {epoch + 1} | "
                f"Policy Loss: {final_actor_result['policy_loss']:.4f} | "
                f"Ratio: {final_actor_result['ratio']:.4f} | "
                f"Clip Fraction: {final_actor_result['clip_fraction']:.3f} | "
                f"Entropy: {final_actor_result['entropy']:.4f} | "
                f"Critic Loss: {final_critic_result['critic_loss']:.4f}"
            )

        return {**final_actor_result, **final_critic_result}
