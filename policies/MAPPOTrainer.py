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
                 ppo_epochs = 10):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

    def update_actor(self, buffer: RolloutBuffer):
        t = buffer.step

        # Process advantages
        advantages = buffer.advantages[:t]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-5)

        actor_losses = []
        entropy_records = []
        ratios = []
        clip_fractions = []

        for i in range(t):
            obs = buffer.observations[i]
            hidden_state = buffer.actor_hidden_states[i]
            mask = buffer.actor_rnn_masks[i]
            actions = buffer.actions[i]
            init_log_probs = buffer.old_log_probs[i]

            # Get the advantage
            advantage = advantages[i]
            expanded_advantage = advantage.expand(buffer.num_agents) # Now use it to all agents

            new_log_probs, entropy, _ = self.actor.evaluate_actions(obs, hidden_state, actions, mask)

            ratio = torch.exp(new_log_probs - init_log_probs)
            unclipped_ratio = ratio * expanded_advantage
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * expanded_advantage

            policy_loss = -torch.min(unclipped_ratio, clipped_ratio).mean()

            actor_losses.append(policy_loss)
            entropy_records.append(entropy.mean())
            ratios.append(ratio.mean())

            clip_fraction = (torch.abs(ratio - 1.0) > self.clip_param).float().mean()
            clip_fractions.append(clip_fraction)

        policy_loss = torch.stack(actor_losses).mean()
        entropy = torch.stack(entropy_records).mean()
        mean_ratio = torch.stack(ratios).mean()
        mean_clip_fraction = torch.stack(clip_fractions).mean()

        # minimize loss
        actor_loss = (policy_loss - self.entropy_coef * entropy)

        # gradient descent
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()

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

        for i in range(t):
            state = buffer.states[i].unsqueeze(0)
            hidden_state = buffer.critic_hidden_states[i]
            mask = buffer.critic_rnn_masks[i]
            target = buffer.returns[i]

            predicted_value, _ = self.critic(state, hidden_state, mask)
            predicted_value = predicted_value.squeeze()

            # Calculate the loss
            #value_loss = torch.nn.functional.mse_loss(predicted_value, target)
            # Instead of MSE the authors of MAPPO use Hubber Loss
            value_loss = torch.nn.functional.huber_loss(predicted_value, target, delta=10.0)
            value_losses.append(value_loss)

        value_loss = torch.stack(value_losses).mean()
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
        results = []
        for epoch in range(self.ppo_epochs):
            actor_result = self.update_actor(buffer)
            critic_result = self.update_critic(buffer)
            result = { **actor_result, **critic_result }
            results.append(result)

            print(
                f"  PPO Epoch {epoch + 1} | "
                f"Policy Loss: {actor_result['policy_loss']:.4f} | "
                f"Ratio: {actor_result['ratio']:.4f} | "
                f"Clip Fraction: {actor_result['clip_fraction']:.3f} | "
                f"Entropy: {actor_result['entropy']:.4f} | "
                f"Critic Loss: {critic_result['critic_loss']:.4f}"
            )

        return results