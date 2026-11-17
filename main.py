
from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from mpe2 import simple_tag_v3

from policies.Actor import Actor
from policies.Critic import Critic
from policies.MAPPOTrainer import MAPPOTrainer
from utils import Utils
from utils.VectorRolloutBuffer import VectorRolloutBuffer
from utils.VectorTagEnv import VectorTagEnv

# Configuration
GLOBAL_SEED = 100
TOTAL_ENV_STEPS = 250_000
NUM_VECTOR_ENVS = 32
MAX_STEPS = 25
ROLLOUT_STEPS = 25
RENDER = False
HIDDEN_DIM = 128
CHECKPOINT_PATH = Path("checkpoints/best_actor.pt")
EVAL_INTERVAL_UPDATES = 10
EVAL_EPISODES = 100
EVAL_SEED = 10_000
TEST_EPISODES = 100
TEST_SEED = 20_000
TEST_OBSERVATION_FAILURE_STEPS = 25  # Set to 0 to disable deployment failure.
SHAPING = True


@dataclass
class VectorCollectorState:
    observations: dict[str, np.ndarray]
    actor_hidden_states: torch.Tensor
    actor_rnn_mask: torch.Tensor
    critic_hidden_states: torch.Tensor
    critic_rnn_mask: torch.Tensor
    episode_returns: np.ndarray
    episode_lengths: np.ndarray


def get_env(render_mode: str | None = None):
    """Create the cooperative pursuit environment."""
    return simple_tag_v3.parallel_env(
        render_mode=render_mode,
        max_cycles=MAX_STEPS,
        terminate_on_success=True,
    )

def set_seed(seed: int, device: torch.device) -> None:
    """Seed the main random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def get_torch_rng_state(device: torch.device):
    """Capture CPU and selected accelerator RNG state for isolated evaluation."""
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state(device)
    return state


def set_torch_rng_state(state, device: torch.device) -> None:
    """Restore CPU and selected accelerator RNG state."""
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])
    elif device.type == "mps":
        torch.mps.set_rng_state(state["mps"], device)


def seed_torch_rng(seed: int, device: torch.device) -> None:
    """Seed the RNG used by a sampled policy on the selected device."""
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def initialize_vector_collector(
    env: gym.vector.VectorEnv,
    seed: int,
    num_agents: int,
    device: torch.device,
) -> VectorCollectorState:
    observations, _ = env.reset(seed=seed)
    return VectorCollectorState(
        observations=observations,
        actor_hidden_states=torch.zeros(
            env.num_envs, num_agents, 1, HIDDEN_DIM, device=device
        ),
        actor_rnn_mask=torch.zeros(
            env.num_envs, num_agents, 1, device=device
        ),
        critic_hidden_states=torch.zeros(
            env.num_envs, 1, HIDDEN_DIM, device=device
        ),
        critic_rnn_mask=torch.zeros(env.num_envs, 1, device=device),
        episode_returns=np.zeros(env.num_envs, dtype=np.float64),
        episode_lengths=np.zeros(env.num_envs, dtype=np.int64),
    )


def collect_vector_rollout(
    env: gym.vector.VectorEnv,
    collector: VectorCollectorState,
    actor: Actor,
    critic: Critic,
    num_agents: int,
    obs_dim: int,
    state_dim: int,
    device: torch.device,
) -> tuple[VectorRolloutBuffer, VectorCollectorState, list[dict]]:
    """Collect one fixed-length rollout from all environments in parallel."""
    buffer = VectorRolloutBuffer(
        ROLLOUT_STEPS,
        env.num_envs,
        num_agents,
        obs_dim,
        state_dim,
        HIDDEN_DIM,
        device,
    )
    completed_episodes = []

    for _ in range(ROLLOUT_STEPS):
        predator_obs = torch.as_tensor(
            collector.observations["predator_observations"],
            dtype=torch.float32,
            device=device,
        )
        global_state = torch.as_tensor(
            collector.observations["state"],
            dtype=torch.float32,
            device=device,
        )
        flat_obs = predator_obs.reshape(env.num_envs * num_agents, obs_dim)
        flat_hidden = collector.actor_hidden_states.reshape(
            env.num_envs * num_agents, 1, HIDDEN_DIM
        )
        flat_mask = collector.actor_rnn_mask.reshape(
            env.num_envs * num_agents, 1
        )

        with torch.no_grad():
            flat_actions, flat_log_probs, flat_next_hidden = actor.act(
                flat_obs,
                flat_hidden,
                flat_mask,
            )
            value, next_critic_hidden = critic(
                global_state,
                collector.critic_hidden_states,
                collector.critic_rnn_mask,
            )

        predator_actions = flat_actions.reshape(env.num_envs, num_agents)
        log_probs = flat_log_probs.reshape(env.num_envs, num_agents)
        next_actor_hidden = flat_next_hidden.reshape(
            env.num_envs, num_agents, 1, HIDDEN_DIM
        )
        next_observations, rewards, terminations, truncations, _ = env.step(
            predator_actions.cpu().numpy()
        )

        done = np.logical_or(terminations, truncations)
        continuing = torch.as_tensor(
            np.logical_not(done), dtype=torch.float32, device=device
        ).unsqueeze(-1)
        next_critic_mask = continuing
        next_actor_mask = continuing.unsqueeze(1).expand(-1, num_agents, -1)
        next_predator_obs = torch.as_tensor(
            next_observations["predator_observations"],
            dtype=torch.float32,
            device=device,
        )
        next_global_state = torch.as_tensor(
            next_observations["state"],
            dtype=torch.float32,
            device=device,
        )
        reward_tensor = torch.as_tensor(
            rewards, dtype=torch.float32, device=device
        )

        buffer.insert(
            predator_obs,
            global_state,
            collector.actor_hidden_states,
            collector.critic_hidden_states,
            collector.actor_rnn_mask,
            collector.critic_rnn_mask,
            predator_actions,
            log_probs,
            value.squeeze(-1),
            reward_tensor,
            next_predator_obs,
            next_global_state,
            next_actor_hidden,
            next_critic_hidden,
            next_actor_mask,
            next_critic_mask,
        )

        collector.episode_returns += rewards
        collector.episode_lengths += 1
        for env_index in np.flatnonzero(done):
            completed_episodes.append({
                "return": float(collector.episode_returns[env_index]),
                "captured": bool(terminations[env_index]),
                "capture_step": (
                    int(collector.episode_lengths[env_index])
                    if terminations[env_index]
                    else None
                ),
            })
        collector.episode_returns[done] = 0.0
        collector.episode_lengths[done] = 0

        collector.observations = next_observations
        collector.actor_hidden_states = next_actor_hidden
        collector.actor_rnn_mask = next_actor_mask
        collector.critic_hidden_states = next_critic_hidden
        collector.critic_rnn_mask = next_critic_mask

    final_state = torch.as_tensor(
        collector.observations["state"],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        next_value, _ = critic(
            final_state,
            collector.critic_hidden_states,
            collector.critic_rnn_mask,
        )
    buffer.compute_returns_and_advantages(next_value.squeeze(-1))
    return buffer, collector, completed_episodes


@torch.no_grad()
def evaluate_policy(
    actor: Actor | None,
    device: torch.device,
    seed: int,
    episodes: int = EVAL_EPISODES,
    action_mode: str = "argmax",
    observation_failure_steps: int = 0,
) -> list[dict[str, float | bool | int | str | None]]:
    """Evaluate a policy, optionally with one predator's observations disabled."""
    if action_mode not in {"argmax", "sample", "random"}:
        raise ValueError(f"Unknown evaluation action mode: {action_mode}")
    if action_mode != "random" and actor is None:
        raise ValueError("An actor is required for argmax and sampled evaluation.")
    if not 0 <= observation_failure_steps <= MAX_STEPS:
        raise ValueError(
            "observation_failure_steps must be between 0 and MAX_STEPS."
        )

    env = get_env()
    actor_was_training = actor.training if actor is not None else None
    if actor is not None:
        actor.eval()
    episode_returns = []
    torch_rng_state = get_torch_rng_state(device)

    try:
        predators = [agent for agent in env.possible_agents if "adversary" in agent]
        prey_agents = [agent for agent in env.possible_agents if "adversary" not in agent]
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            failed_agent = None
            failure_start = None
            failure_end = None
            if observation_failure_steps > 0 and action_mode != "random":
                failure_rng = random.Random(episode_seed)
                failed_agent = failure_rng.choice(predators)
                failure_start = failure_rng.randint(
                    0,
                    MAX_STEPS - observation_failure_steps,
                )
                failure_end = failure_start + observation_failure_steps

            # Make sampled-policy evaluation repeatable without changing training RNG state.
            seed_torch_rng(episode_seed, device)
            observations, _ = env.reset(seed=episode_seed)
            for agent_index, agent in enumerate(env.possible_agents):
                env.action_space(agent).seed(episode_seed + agent_index + 1)

            hidden_state = Utils.init_hidden_state(len(predators), HIDDEN_DIM)
            mask = Utils.init_rnn_mask(len(predators), 1)
            episode_return = 0.0
            capture_step = None

            for step_index in range(MAX_STEPS):
                if not env.agents:
                    break

                active_predators = [agent for agent in predators if agent in env.agents]
                predator_obs = torch.as_tensor(
                    np.stack([observations[agent] for agent in active_predators]),
                    dtype=torch.float32,
                    device=device,
                )
                failure_active = (
                    failed_agent in active_predators
                    and failure_start <= step_index < failure_end
                )
                if failure_active:
                    failed_index = active_predators.index(failed_agent)
                    predator_obs[failed_index].zero_()

                actions = {
                    agent: env.action_space(agent).sample()
                    for agent in env.agents
                    if agent not in predators
                }
                if action_mode == "random":
                    actions.update({
                        agent: env.action_space(agent).sample()
                        for agent in active_predators
                    })
                else:
                    dist, next_hidden_state = actor(predator_obs, hidden_state, mask)
                    if action_mode == "argmax":
                        predator_actions = dist.probs.argmax(dim=-1)
                    else:
                        predator_actions = dist.sample()
                    for index, agent in enumerate(active_predators):
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
                "failed_agent": failed_agent,
                "observation_failure_start": failure_start,
                "observation_failure_end": failure_end,
            })
    finally:
        env.close()
        set_torch_rng_state(torch_rng_state, device)
        if actor is not None:
            actor.train(actor_was_training)

    return episode_returns


def train(seed: int, device: torch.device) -> dict[str, dict[str, float]]:
    """Train once and save the best evaluated actor."""
    set_seed(seed, device)
    if RENDER:
        raise ValueError("Human rendering is not supported during vectorized training.")

    env = AsyncVectorEnv(
        [
            partial(
                VectorTagEnv,
                max_steps=MAX_STEPS,
                shaping=SHAPING,
                terminate_on_success=True,
            )
            for _ in range(NUM_VECTOR_ENVS)
        ],
        shared_memory=True,
        context="spawn",
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    eval_seed = EVAL_SEED

    try:
        predator_space = env.single_observation_space["predator_observations"]
        obs_dim = predator_space.shape[-1]
        predator_count = predator_space.shape[0]
        state_dim = env.single_observation_space["state"].shape[0]
        action_dim = int(env.single_action_space.nvec[0])

        actor = Actor(obs_dim, HIDDEN_DIM, action_dim).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=5e-4, eps=1e-5)
        critic = Critic(state_dim, HIDDEN_DIM).to(device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=5e-4, eps=1e-5)
        trainer = MAPPOTrainer(
            actor, actor_optimizer, critic, critic_optimizer, ppo_epochs=5
        )

        best_eval_mean = float("-inf")
        first_capture_episode = None
        recent_captures = deque(maxlen=100)
        episode_index = 0
        collector = initialize_vector_collector(
            env,
            seed,
            predator_count,
            device,
        )
        target_env_steps = TOTAL_ENV_STEPS
        steps_per_update = NUM_VECTOR_ENVS * ROLLOUT_STEPS
        num_updates = (target_env_steps + steps_per_update - 1) // steps_per_update

        for update_zero_indexed in range(num_updates):
            update_index = update_zero_indexed + 1
            buffer, collector, completed_episodes = collect_vector_rollout(
                env,
                collector,
                actor,
                critic,
                predator_count,
                obs_dim,
                state_dim,
                device,
            )
            trainer.update(buffer)

            batch_captures = sum(
                episode["captured"]
                for episode in completed_episodes
            )
            for episode in completed_episodes:
                episode_index += 1
                recent_captures.append(episode["captured"])
                if episode["captured"] and first_capture_episode is None:
                    first_capture_episode = episode_index

            batch_capture_rate = (
                batch_captures / len(completed_episodes)
                if completed_episodes
                else float("nan")
            )
            recent_capture_rate = (
                float(np.mean(recent_captures))
                if recent_captures
                else float("nan")
            )
            total_env_steps = update_index * steps_per_update
            print(
                f"Seed {seed} | Update {update_index}/{num_updates} | "
                f"Env steps: {total_env_steps} | Episodes completed: {episode_index} | "
                f"Batch capture rate: {batch_capture_rate:.0%} "
                f"({batch_captures}/{len(completed_episodes)}) | "
                f"Recent capture rate (last {len(recent_captures)}): "
                f"{recent_capture_rate:.0%} | "
                f"First capture episode: {first_capture_episode or 'none'}"
            )

            should_evaluate = (
                update_index % EVAL_INTERVAL_UPDATES == 0
                or update_index == num_updates
            )
            if should_evaluate:
                eval_episodes = evaluate_policy(
                    actor,
                    device,
                    eval_seed,
                    EVAL_EPISODES,
                )
                eval_returns = [episode["return"] for episode in eval_episodes]
                eval_mean = float(np.mean(eval_returns))
                eval_std = float(np.std(eval_returns))
                eval_capture_rate = float(np.mean([
                    episode["captured"] for episode in eval_episodes
                ]))
                print(
                    f"Seed {seed} | Update {update_index} | Episode {episode_index} | "
                    f"Eval return: {eval_mean:.2f} +/- {eval_std:.2f} | "
                    f"Capture rate: {eval_capture_rate:.0%}"
                )

                if eval_mean > best_eval_mean:
                    best_eval_mean = eval_mean
                    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "actor_state_dict": actor.state_dict(),
                            "best_eval_mean": best_eval_mean,
                            "best_eval_std": eval_std,
                            "evaluation_returns": eval_returns,
                            "eval_capture_rate": eval_capture_rate,
                            "training_seed": seed,
                            "episode": episode_index,
                            "update": update_index,
                            "config": {
                                "target_env_steps": target_env_steps,
                                "num_vector_envs": NUM_VECTOR_ENVS,
                                "rollout_steps": ROLLOUT_STEPS,
                                "num_updates": num_updates,
                                "max_steps": MAX_STEPS,
                                "shaping": SHAPING,
                                "hidden_dim": HIDDEN_DIM,
                                "ppo_epochs": trainer.ppo_epochs,
                                "clip_param": trainer.clip_param,
                                "entropy_coef": trainer.entropy_coef,
                                "gamma": 0.99,
                                "gae_lambda": 0.95,
                                "learning_rate": 5e-4,
                                "eval_seed": eval_seed,
                                "eval_episodes": EVAL_EPISODES,
                                "eval_policy": "deterministic_argmax",
                                "test_seed": TEST_SEED,
                                "test_episodes": TEST_EPISODES,
                                "recurrent_chunk_length": trainer.recurrent_chunk_length,
                            },
                        },
                        CHECKPOINT_PATH,
                    )
                    print(f"Saved best evaluated actor to {CHECKPOINT_PATH}")

        best_checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        actor.load_state_dict(best_checkpoint["actor_state_dict"])
        return {
            "test_results": test_checkpoint(actor, device),
        }
    finally:
        env.close()


def test_checkpoint(actor: Actor, device: torch.device) -> dict[str, float]:
    """Compare clean and observation-masked deployment on held-out seeds."""
    print(
        f"Testing on {TEST_EPISODES} held-out episodes "
        f"(seeds {TEST_SEED}–{TEST_SEED + TEST_EPISODES - 1})..."
    )
    results = {}
    conditions = (
        ("No observation masking", 0),
        (
            "Observation masking: one random predator for "
            f"{TEST_OBSERVATION_FAILURE_STEPS} steps",
            TEST_OBSERVATION_FAILURE_STEPS,
        ),
    )

    for condition_label, failure_steps in conditions:
        print(f"\n{condition_label}:")
        for policy_label, mode in (
            ("PPO deterministic", "argmax"),
            ("PPO sampled", "sample"),
        ):
            episodes = evaluate_policy(
                actor,
                device,
                TEST_SEED,
                TEST_EPISODES,
                action_mode=mode,
                observation_failure_steps=failure_steps,
            )
            returns = [episode["return"] for episode in episodes]
            mean = float(np.mean(returns))
            std = float(np.std(returns))
            capture_rate = float(np.mean([
                episode["captured"] for episode in episodes
            ]))
            capture_steps = [
                episode["capture_step"]
                for episode in episodes
                if episode["captured"]
            ]
            mean_capture_step = (
                float(np.mean(capture_steps))
                if capture_steps
                else float("nan")
            )
            results[f"{condition_label} | {policy_label}"] = mean
            print(
                f"{policy_label}: return {mean:.2f} +/- {std:.2f} | "
                f"captures {capture_rate:.0%} | "
                f"mean capture step {mean_capture_step:.1f}"
            )

    print("\nRandom baseline:")
    random_episodes = evaluate_policy(
        None,
        device,
        TEST_SEED,
        TEST_EPISODES,
        action_mode="random",
    )
    random_returns = [episode["return"] for episode in random_episodes]
    random_mean = float(np.mean(random_returns))
    random_std = float(np.std(random_returns))
    random_capture_rate = float(np.mean([
        episode["captured"] for episode in random_episodes
    ]))
    random_capture_steps = [
        episode["capture_step"]
        for episode in random_episodes
        if episode["captured"]
    ]
    random_mean_capture_step = (
        float(np.mean(random_capture_steps))
        if random_capture_steps
        else float("nan")
    )
    results["Random"] = random_mean
    print(
        f"Random: return {random_mean:.2f} +/- {random_std:.2f} | "
        f"captures {random_capture_rate:.0%} | "
        f"mean capture step {random_mean_capture_step:.1f}"
    )
    return results


def test_saved_checkpoint(device: torch.device) -> dict[str, float]:
    """Load the saved actor and test it without running training."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT_PATH}. Train first or set CHECKPOINT_PATH."
        )

    env = get_env()
    try:
        sample_predator_id = "adversary_0"
        observation_dim = env.observation_space(sample_predator_id).shape[0]
        action_dim = env.action_space(sample_predator_id).n
        actor = Actor(observation_dim, HIDDEN_DIM, action_dim).to(device)
    finally:
        env.close()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    return test_checkpoint(actor, device)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train or test the MAPPO policy.")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Test checkpoints/best_actor.pt without running training.",
    )
    args = parser.parse_args()

    device = Utils.get_torch_device()
    print(f"PyTorch device: {device}")
    if args.test_only:
        test_saved_checkpoint(device)
        return

    print(
        f"Running {TOTAL_ENV_STEPS} MAPPO environment steps "
        f"across {NUM_VECTOR_ENVS} vector environments with seed {GLOBAL_SEED}..."
    )
    train(GLOBAL_SEED, device)


if __name__ == "__main__":
    main()
