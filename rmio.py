"""Core RMIO model, training, and observation-loss inference components."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import platform
from typing import Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F


def select_device() -> torch.device:
    """Prefer MPS on macOS and CUDA on Linux, falling back to CPU."""
    system = platform.system()
    if system == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if system == "Linux" and torch.cuda.is_available():
        return torch.device("cuda")
    if system not in {"Darwin", "Linux"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mlp(input_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_size, hidden_size), nn.ELU(),
                         nn.Linear(hidden_size, hidden_size), nn.ELU(),
                         nn.Linear(hidden_size, output_size))


class WorldModelOutput(NamedTuple):
    observation: Tensor
    reward: Tensor
    continuation: Tensor
    prior_mean: Tensor
    prior_log_std: Tensor
    posterior_mean: Tensor
    posterior_log_std: Tensor
    hidden: Tensor
    latent: Tensor


class MultiAgentWorldModel(nn.Module):
    """Cross-agent RSSM. Tensors use [batch, agents, feature] dimensions."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 128,
                 latent_size: int = 32, attention_heads: int = 4):
        super().__init__()
        self.observation_size, self.action_size = observation_size, action_size
        self.hidden_size, self.latent_size = hidden_size, latent_size
        self.observation_encoder = mlp(observation_size, hidden_size, hidden_size)
        self.action_encoder = mlp(action_size, hidden_size, hidden_size)
        self.latent_encoder = mlp(latent_size, hidden_size, hidden_size)
        self.communication_encoder = mlp(hidden_size * 2, hidden_size, hidden_size)
        self.communication = nn.MultiheadAttention(hidden_size, attention_heads, batch_first=True)
        self.recurrent = nn.GRUCell(hidden_size, hidden_size)
        self.prior = mlp(hidden_size, hidden_size, latent_size * 2)
        self.posterior = mlp(hidden_size * 2, hidden_size, latent_size * 2)
        feature_size = hidden_size + latent_size
        self.observation_head = mlp(feature_size, hidden_size, observation_size)
        self.reward_head = mlp(feature_size, hidden_size, 1)
        self.continuation_head = mlp(feature_size, hidden_size, 1)

    def initial_state(self, batch_size: int, n_agents: int, *, device=None) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(batch_size, n_agents, self.hidden_size, device=device),
                torch.zeros(batch_size, n_agents, self.latent_size, device=device))

    @staticmethod
    def _sample(mean: Tensor, log_std: Tensor, sample: bool) -> Tensor:
        return mean + log_std.exp() * torch.randn_like(mean) if sample else mean

    def _transition(self, actions: Tensor, hidden: Tensor, latent: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        action_embedding = self.action_encoder(actions)
        latent_embedding = self.latent_encoder(latent)
        message = self.communication_encoder(torch.cat((latent_embedding, action_embedding), dim=-1))
        message, _ = self.communication(message, message, message, need_weights=False)
        next_hidden = self.recurrent(message.flatten(0, 1), hidden.flatten(0, 1)).view_as(hidden)
        mean, log_std = self.prior(next_hidden).chunk(2, dim=-1)
        return next_hidden, mean, log_std.clamp(-7, 2)

    def step(self, observations: Tensor, actions_previous: Tensor, hidden: Tensor,
             previous_latent: Tensor, *, sample: bool = True) -> WorldModelOutput:
        """Posterior update/reconstruction from the current observation."""
        next_hidden, prior_mean, prior_log_std = self._transition(actions_previous, hidden, previous_latent)
        obs_embedding = self.observation_encoder(observations)
        post_mean, post_log_std = self.posterior(torch.cat((next_hidden, obs_embedding), dim=-1)).chunk(2, dim=-1)
        post_log_std = post_log_std.clamp(-7, 2)
        latent = self._sample(post_mean, post_log_std, sample)
        features = torch.cat((next_hidden, latent), dim=-1)
        return WorldModelOutput(self.observation_head(features), self.reward_head(features).squeeze(-1),
                                self.continuation_head(features).squeeze(-1).sigmoid(), prior_mean,
                                prior_log_std, post_mean, post_log_std, next_hidden, latent)

    def imagine_step(self, actions: Tensor, hidden: Tensor, previous_latent: Tensor,
                     *, sample: bool = True) -> WorldModelOutput:
        """Prior-only transition to predict observations, rewards, and continuation."""
        next_hidden, mean, log_std = self._transition(actions, hidden, previous_latent)
        latent = self._sample(mean, log_std, sample)
        features = torch.cat((next_hidden, latent), dim=-1)
        return WorldModelOutput(self.observation_head(features), self.reward_head(features).squeeze(-1),
                                self.continuation_head(features).squeeze(-1).sigmoid(), mean, log_std,
                                mean, log_std, next_hidden, latent)

    @staticmethod
    def loss(output: WorldModelOutput, target_observation: Tensor, target_reward: Tensor,
             target_continuation: Tensor, kl_weight: float = 1.0) -> Dict[str, Tensor]:
        obs_loss = F.mse_loss(output.observation, target_observation)
        reward_loss = F.mse_loss(output.reward, target_reward)
        continuation_loss = F.binary_cross_entropy(output.continuation, target_continuation.float())
        prior = Normal(output.prior_mean, output.prior_log_std.exp())
        posterior = Normal(output.posterior_mean, output.posterior_log_std.exp())
        kl = torch.distributions.kl_divergence(posterior, prior).mean()
        total = obs_loss + reward_loss + continuation_loss + kl_weight * kl
        return {"loss": total, "observation": obs_loss, "reward": reward_loss,
                "continuation": continuation_loss, "kl": kl}


class ObservationCorrector(nn.Module):
    """Cross-agent attention block that corrects imputed observations."""

    def __init__(self, observation_size: int, hidden_size: int = 128, attention_heads: int = 4):
        super().__init__()
        self.encoder = mlp(observation_size * 2 + 1, hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, attention_heads, batch_first=True)
        self.decoder = mlp(hidden_size, hidden_size, observation_size)

    def forward(self, observations: Tensor, available: Tensor) -> Tensor:
        mask = available.to(observations.dtype).unsqueeze(-1)
        tokens = self.encoder(torch.cat((observations * mask, observations, mask), dim=-1))
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        estimates = self.decoder(tokens + attended)
        return torch.where(available.unsqueeze(-1), observations, estimates)

    def loss(self, observations: Tensor, available: Tensor, targets: Tensor) -> Tensor:
        estimate = self(observations, available)
        missing = (~available).unsqueeze(-1).expand_as(targets)
        if not missing.any():
            return estimate.sum() * 0
        return F.mse_loss(estimate[missing], targets[missing])


class RecurrentActorCritic(nn.Module):
    """Shared per-agent recurrent actor and centralized critic."""

    def __init__(self, observation_size: int, action_size: int, n_agents: int,
                 hidden_size: int = 128, continuous: bool = False):
        super().__init__()
        self.action_size, self.n_agents, self.continuous = action_size, n_agents, continuous
        self.encoder = mlp(observation_size, hidden_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size)) if continuous else None
        self.critic = mlp(observation_size * n_agents, hidden_size, 1)

    def initial_state(self, batch_size: int, *, device=None) -> Tensor:
        return torch.zeros(batch_size, self.n_agents, self.gru.hidden_size, device=device)

    def forward(self, observations: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch, agents, _ = observations.shape
        encoded = self.encoder(observations).reshape(batch * agents, -1)
        next_hidden = self.gru(encoded, hidden.reshape(batch * agents, -1)).view(batch, agents, -1)
        return self.actor(next_hidden), self.critic(observations.flatten(1)).squeeze(-1), next_hidden

    def _distribution(self, params: Tensor):
        if self.continuous:
            return Normal(params, self.log_std.exp().expand_as(params))
        return Categorical(logits=params)

    def act(self, observations: Tensor, hidden: Tensor, deterministic: bool = False):
        params, value, next_hidden = self(observations, hidden)
        dist = self._distribution(params)
        if deterministic:
            actions = dist.mean if self.continuous else params.argmax(-1)
        else:
            actions = dist.sample()
        log_prob = dist.log_prob(actions).sum(-1) if self.continuous else dist.log_prob(actions)
        return actions, log_prob, value, next_hidden

    def evaluate_actions(self, observations: Tensor, hidden: Tensor, actions: Tensor):
        params, value, next_hidden = self(observations, hidden)
        dist = self._distribution(params)
        log_prob = dist.log_prob(actions).sum(-1) if self.continuous else dist.log_prob(actions)
        entropy = dist.entropy().sum(-1) if self.continuous else dist.entropy()
        return log_prob, entropy, value, next_hidden


@dataclass
class Transition:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    dones: Tensor
    log_probs: Optional[Tensor] = None
    values: Optional[Tensor] = None
    next_value: Optional[Tensor] = None


class TrajectoryReplay:
    """Bounded episode replay. Use separate instances for real and imagined data."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.episodes: Deque[List[Transition]] = deque(maxlen=capacity)

    def add_episode(self, episode: Sequence[Transition]) -> None:
        if episode:
            self.episodes.append(list(episode))

    def sample(self, batch_size: int) -> List[Transition]:
        if not self.episodes:
            raise ValueError("cannot sample an empty replay buffer")
        flat = [step for episode in self.episodes for step in episode]
        indices = torch.randint(len(flat), (min(batch_size, len(flat)),)).tolist()
        return [flat[index] for index in indices]

    def __len__(self) -> int:
        return sum(map(len, self.episodes))


def smooth_rewards(rewards: Tensor, radius: int = 3, sigma: float = 1.0) -> Tensor:
    """Apply normalized Gaussian smoothing along the last (time) dimension."""
    if radius < 0 or sigma <= 0:
        raise ValueError("radius must be nonnegative and sigma positive")
    if radius == 0:
        return rewards
    offsets = torch.arange(-radius, radius + 1, device=rewards.device, dtype=rewards.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = kernel / kernel.sum()
    flat = rewards.reshape(-1, 1, rewards.shape[-1])
    padded = F.pad(flat, (radius, radius), mode="replicate")
    return F.conv1d(padded, kernel.view(1, 1, -1)).reshape_as(rewards)


def generalized_advantage_estimate(rewards: Tensor, values: Tensor, dones: Tensor,
                                   gamma: float = 0.99, lam: float = 0.95,
                                   bootstrap_value: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Calculate GAE advantages and returns for [time, ...] trajectories."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(values[-1])
    next_value = torch.zeros_like(values[-1]) if bootstrap_value is None else bootstrap_value
    for t in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[t].to(rewards.dtype)
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        carry = delta + gamma * lam * nonterminal * carry
        advantages[t] = carry
        next_value = values[t]
    return advantages, advantages + values


def ppo_loss(new_log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor,
             values: Tensor, returns: Tensor, entropy: Tensor, clip_ratio: float = 0.2,
             value_coefficient: float = 0.5, entropy_coefficient: float = 0.01) -> Dict[str, Tensor]:
    ratio = (new_log_probs - old_log_probs).exp()
    while advantages.ndim < ratio.ndim:
        advantages = advantages.unsqueeze(-1)
    policy_loss = -torch.minimum(ratio * advantages,
                                 ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_mean = entropy.mean()
    return {"loss": policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_mean,
            "policy": policy_loss, "value": value_loss, "entropy": entropy_mean}


class RMIOInference:
    """Inference controller; world-model synchronization occurs on observation loss."""

    def __init__(self, world_model: MultiAgentWorldModel, corrector: ObservationCorrector,
                 policy: RecurrentActorCritic, history_horizon: int = 8,
                 device: Optional[torch.device | str] = None):
        if history_horizon <= 0:
            raise ValueError("history_horizon must be positive")
        self.world_model, self.corrector, self.policy = world_model, corrector, policy
        self.device = torch.device(device) if device is not None else select_device()
        self.world_model.to(self.device)
        self.corrector.to(self.device)
        self.policy.to(self.device)
        self.history_horizon = history_horizon
        self.reset(1, policy.n_agents)

    def reset(self, batch_size: int = 1, n_agents: Optional[int] = None, device=None) -> None:
        n_agents = n_agents or self.policy.n_agents
        device = self.device if device is None else device
        self.hidden, self.latent = self.world_model.initial_state(batch_size, n_agents, device=device)
        self.policy_hidden = self.policy.initial_state(batch_size, device=device)
        self.history: Deque[Tuple[Tensor, Tensor]] = deque(maxlen=self.history_horizon)
        self.history_truncated = False

    @torch.no_grad()
    def act(self, observations: Tensor, actions_previous: Optional[Tensor], available: Tensor,
            deterministic: bool = True):
        observations = observations.to(self.device).float()
        batch_size, n_agents = observations.shape[:2]
        if actions_previous is None:
            action_features = torch.zeros(batch_size, n_agents, self.world_model.action_size,
                                          device=self.device)
        elif self.policy.continuous or actions_previous.ndim == 3:
            action_features = actions_previous.to(self.device).float()
        else:
            action_features = F.one_hot(actions_previous.to(self.device).long(),
                                        self.policy.action_size).float()
        available = available.to(device=self.device, dtype=torch.bool)
        if bool(available.all()):
            completed = observations
            if len(self.history) == self.history_horizon:
                self.history_truncated = True
            self.history.append((observations.detach(), action_features.detach()))
        else:
            if self.history_truncated:
                self.hidden, self.latent = self.world_model.initial_state(
                    batch_size, n_agents, device=self.device)
            for past_observations, past_actions in self.history:
                sync = self.world_model.step(past_observations, past_actions,
                                             self.hidden, self.latent, sample=False)
                self.hidden, self.latent = sync.hidden, sync.latent
            self.history.clear()
            self.history_truncated = False
            prior = self.world_model.imagine_step(action_features, self.hidden, self.latent, sample=False)
            candidate = torch.where(available.unsqueeze(-1), observations, prior.observation)
            completed = self.corrector(candidate, available)
            out = self.world_model.step(completed, action_features, self.hidden, self.latent, sample=False)
            self.hidden, self.latent = out.hidden, out.latent
        actions, log_probs, values, self.policy_hidden = self.policy.act(
            completed, self.policy_hidden, deterministic)
        return actions, log_probs, values, completed


@dataclass
class RMIOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    kl_weight: float = 1.0
    rollout_steps: int = 15
    model_epochs: int = 1
    policy_epochs: int = 4
    correction_epochs: int = 1
    reward_smoothing_radius: int = 3
    reward_smoothing_sigma: float = 1.0
    observation_loss_probability: float = 0.25
    max_grad_norm: float = 10.0


class RMIOTrainer:
    """Environment-agnostic alternating RMIO training loop.

    Real episodes contain ``Transition`` entries with observations/actions shaped
    [agents, features], a team reward, next observations, and a termination flag.
    """

    def __init__(self, world_model: MultiAgentWorldModel, corrector: ObservationCorrector,
                 policy: RecurrentActorCritic, config: Optional[RMIOConfig] = None,
                 learning_rate: float = 3e-4, real_capacity: int = 1000,
                 imagined_capacity: int = 1000, device: Optional[torch.device | str] = None):
        self.world_model, self.corrector, self.policy = world_model, corrector, policy
        self.device = torch.device(device) if device is not None else select_device()
        self.world_model.to(self.device)
        self.corrector.to(self.device)
        self.policy.to(self.device)
        self.config = config or RMIOConfig()
        self.real_replay = TrajectoryReplay(real_capacity)
        self.imagined_replay = TrajectoryReplay(imagined_capacity)
        self.model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=learning_rate)
        self.corrector_optimizer = torch.optim.Adam(self.corrector.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def add_real_episode(self, episode: Sequence[Transition]) -> None:
        self.real_replay.add_episode(episode)

    def _action_features(self, actions: Tensor) -> Tensor:
        return actions.float() if self.policy.continuous else F.one_hot(
            actions.long(), self.policy.action_size).float()

    @staticmethod
    def _scalar(value: Tensor, device: torch.device) -> Tensor:
        return torch.as_tensor(value, device=device, dtype=torch.float32).mean()

    def update_world_model(self, episodes: Optional[Sequence[Sequence[Transition]]] = None) -> Dict[str, float]:
        episodes = episodes or list(self.real_replay.episodes)
        if not episodes:
            raise ValueError("world-model update requires real episodes")
        metrics: Dict[str, float] = {}
        for _ in range(self.config.model_epochs):
            self.model_optimizer.zero_grad(set_to_none=True)
            losses, components = [], {}
            for episode in episodes:
                if not episode:
                    continue
                hidden, latent = self.world_model.initial_state(1, self.policy.n_agents, device=self.device)
                previous_actions = torch.zeros(1, self.policy.n_agents,
                                               self.world_model.action_size, device=self.device)
                rewards = smooth_rewards(torch.stack([self._scalar(x.rewards, self.device) for x in episode]),
                                         self.config.reward_smoothing_radius,
                                         self.config.reward_smoothing_sigma)
                for index, transition in enumerate(episode):
                    obs = transition.observations.to(self.device).float().unsqueeze(0)
                    out = self.world_model.step(obs, previous_actions, hidden, latent, sample=True)
                    done = float(torch.as_tensor(transition.dones).float().max())
                    continuation = torch.full_like(out.continuation, 1.0 - done)
                    target_reward = rewards[index].expand_as(out.reward)
                    loss_items = self.world_model.loss(out, obs, target_reward, continuation,
                                                       self.config.kl_weight)
                    losses.append(loss_items["loss"] / len(episode))
                    for name, value in loss_items.items():
                        components.setdefault(name, []).append(value.detach())
                    hidden, latent = out.hidden, out.latent
                    previous_actions = self._action_features(transition.actions.to(self.device)).unsqueeze(0)
            if not losses:
                raise ValueError("world-model update received only empty episodes")
            total = torch.stack(losses).sum()
            total.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.max_grad_norm)
            self.model_optimizer.step()
            metrics = {k: torch.stack(v).mean().item() for k, v in components.items()}
        return metrics

    @torch.no_grad()
    def imagine_episode(self, seed: Transition, horizon: Optional[int] = None) -> List[Transition]:
        horizon = horizon or self.config.rollout_steps
        obs = seed.observations.to(self.device).float().unsqueeze(0)
        hidden, latent = self.world_model.initial_state(1, self.policy.n_agents, device=self.device)
        prev_action = torch.zeros(1, self.policy.n_agents, self.world_model.action_size, device=self.device)
        encoded = self.world_model.step(obs, prev_action, hidden, latent, sample=False)
        hidden, latent = encoded.hidden, encoded.latent
        policy_hidden = self.policy.initial_state(1, device=self.device)
        current = obs
        episode = []
        for _ in range(horizon):
            actions, logp, value, policy_hidden = self.policy.act(current, policy_hidden)
            action_features = self._action_features(actions).unsqueeze(0)
            prediction = self.world_model.imagine_step(action_features, hidden, latent, sample=True)
            next_obs = prediction.observation
            _, next_value, _ = self.policy(next_obs, policy_hidden)
            done = prediction.continuation.mean() < 0.5
            episode.append(Transition(current.squeeze(0).detach(), actions.squeeze(0).detach(),
                                      prediction.reward.mean().detach(), next_obs.squeeze(0).detach(),
                                      done.detach(), logp.squeeze(0).detach(), value.squeeze(0).detach(),
                                      next_value.squeeze(0).detach()))
            current, hidden, latent = next_obs, prediction.hidden, prediction.latent
            if bool(done):
                break
        return episode

    def update_policy(self, episodes: Optional[Sequence[Sequence[Transition]]] = None) -> Dict[str, float]:
        episodes = episodes or list(self.imagined_replay.episodes)
        if not episodes:
            raise ValueError("policy update requires imagined episodes")
        metrics: Dict[str, float] = {}
        for _ in range(self.config.policy_epochs):
            self.policy_optimizer.zero_grad(set_to_none=True)
            epoch_losses, components = [], {}
            for episode in episodes:
                if not episode:
                    continue
                if any(x.log_probs is None or x.values is None for x in episode):
                    raise ValueError("imagined transitions need behavior log probabilities and values")
                obs = torch.stack([x.observations for x in episode]).to(self.device).float()
                actions = torch.stack([x.actions for x in episode]).to(self.device)
                old_logp = torch.stack([x.log_probs for x in episode]).to(self.device)
                old_values = torch.stack([x.values for x in episode]).to(self.device).reshape(-1)
                rewards = torch.stack([self._scalar(x.rewards, self.device) for x in episode])
                dones = torch.stack([torch.as_tensor(x.dones, device=self.device).float().max() for x in episode])
                with torch.no_grad():
                    bootstrap = episode[-1].next_value
                    if bootstrap is None:
                        h = self.policy.initial_state(1, device=self.device)
                        for item in obs:
                            _, _, h = self.policy(item.unsqueeze(0), h)
                        _, bootstrap, _ = self.policy(episode[-1].next_observations.to(self.device).float().unsqueeze(0), h)
                    else:
                        bootstrap = bootstrap.to(self.device).reshape(1)
                    advantage, returns = generalized_advantage_estimate(
                        rewards, old_values, dones, self.config.gamma, self.config.gae_lambda,
                        bootstrap.squeeze(0))
                    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                hidden = self.policy.initial_state(1, device=self.device)
                logps, entropies, values = [], [], []
                for observation, action in zip(obs, actions):
                    logp, entropy, value, hidden = self.policy.evaluate_actions(
                        observation.unsqueeze(0), hidden, action.unsqueeze(0))
                    logps.append(logp.squeeze(0))
                    entropies.append(entropy.squeeze(0).mean())
                    values.append(value.squeeze(0))
                losses = ppo_loss(torch.stack(logps), old_logp, advantage,
                                  torch.stack(values), returns, torch.stack(entropies), self.config.ppo_clip)
                epoch_losses.append(losses["loss"] / len(episodes))
                for name, value in losses.items():
                    components.setdefault(name, []).append(value.detach())
            total = torch.stack(epoch_losses).sum()
            total.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()
            metrics = {k: torch.stack(v).mean().item() for k, v in components.items()}
        return metrics

    def update_corrector(self, episodes: Optional[Sequence[Sequence[Transition]]] = None) -> Dict[str, float]:
        episodes = episodes or list(self.real_replay.episodes)
        if not episodes:
            raise ValueError("corrector update requires real episodes")
        metric = 0.0
        for _ in range(self.config.correction_epochs):
            self.corrector_optimizer.zero_grad(set_to_none=True)
            losses = []
            for episode in episodes:
                hidden, latent = self.world_model.initial_state(1, self.policy.n_agents, device=self.device)
                prev_action = torch.zeros(1, self.policy.n_agents, self.world_model.action_size, device=self.device)
                for transition in episode:
                    target = transition.observations.to(self.device).float().unsqueeze(0)
                    with torch.no_grad():
                        prior = self.world_model.imagine_step(prev_action, hidden, latent, sample=False)
                    available = torch.rand(1, self.policy.n_agents, device=self.device) >= self.config.observation_loss_probability
                    if bool(available.all()):
                        available[0, torch.randint(self.policy.n_agents, (1,)).item()] = False
                    elif bool((~available).all()) and self.policy.n_agents > 1:
                        available[0, torch.randint(self.policy.n_agents, (1,)).item()] = True
                    candidate = torch.where(available.unsqueeze(-1), target, prior.observation)
                    losses.append(self.corrector.loss(candidate, available, target) / len(episode))
                    corrected = self.corrector(candidate, available)
                    with torch.no_grad():
                        posterior = self.world_model.step(corrected, prev_action, hidden, latent, sample=False)
                        hidden, latent = posterior.hidden, posterior.latent
                    prev_action = self._action_features(transition.actions.to(self.device)).unsqueeze(0)
            if not losses:
                raise ValueError("corrector update received only empty episodes")
            total = torch.stack(losses).sum()
            total.backward()
            nn.utils.clip_grad_norm_(self.corrector.parameters(), self.config.max_grad_norm)
            self.corrector_optimizer.step()
            metric = total.detach().item()
        return {"correction": metric}

    def train_iteration(self) -> Dict[str, Dict[str, float]]:
        """Update all three RMIO components after collecting real episodes."""
        if not self.real_replay.episodes:
            raise ValueError("add real episodes before training")
        world_metrics = self.update_world_model()
        seeds = [ep[torch.randint(len(ep), (1,)).item()]
                 for ep in self.real_replay.episodes if ep]
        for seed in seeds:
            self.imagined_replay.add_episode(self.imagine_episode(seed))
        policy_metrics = self.update_policy()
        correction_metrics = self.update_corrector()
        return {"world_model": world_metrics, "policy": policy_metrics,
                "corrector": correction_metrics}
