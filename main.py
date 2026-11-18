
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from policies.Actor import Actor
from policies.Critic import Critic
from policies.MAPPOTrainer import MAPPOTrainer
from utils import Utils
from utils.EnvUtils import get_env, set_seed
from utils.RolloutBuffer import RolloutBuffer

import os
from dotenv import load_dotenv

# Configuration
load_dotenv("config.env")

GLOBAL_SEED=int(os.getenv("GLOBAL_SEED", np.nan))
NUM_EPISODES=int(os.getenv("NUM_EPISODES", np.nan))
ROLLOUT_EPISODES=int(os.getenv("ROLLOUT_EPISODES", np.nan))
MAX_STEPS=int(os.getenv("MAX_STEPS", np.nan))
RENDER=os.getenv("RENDER", "False").strip().lower() == "true"
HIDDEN_DIM=int(os.getenv("HIDDEN_DIM", np.nan))
CHECKPOINT_PATH=Path(os.getenv("CHECKPOINT_PATH", "."))
EVAL_INTERVAL=int(os.getenv("EVAL_INTERVAL", np.nan))
EVAL_EPISODES=int(os.getenv("EVAL_EPISODES", np.nan))
EVAL_SEED=int(os.getenv("EVAL_SEED", np.nan))
TEST_EPISODES=int(os.getenv("TEST_EPISODES", np.nan))
TEST_SEED=int(os.getenv("TEST_SEED", np.nan))
SHAPING=os.getenv("SHAPING", "True").strip().lower() == "true"
TEST_OBSERVATION_FAILURE_STEPS=int(os.getenv("TEST_OBSERVATION_FAILURE_STEPS", np.nan))


def run_episode(env,
                episode_seed: int,
                actor: Actor,
                critic: Critic,
                buffer: RolloutBuffer,
                device: torch.device) -> dict:
    """Run one episode and return the cumulative mean predator reward."""
    # Give every episode a distinct, reproducible environment seed.
    observations, infos = env.reset(seed=episode_seed)
    episode_reward = 0.0
    predator_agents = []
    prey_agents = []

    # Seed each action space for reproducible random actions.
    for agent_index, agent in enumerate(env.possible_agents):
        env.action_space(agent).seed(episode_seed + agent_index + 1)
        if "adversary" in agent:
            predator_agents.append(agent)
        else:
            prey_agents.append(agent)

    # The batch size corresponds to the number of agents to train
    actor_hidden_states = Utils.init_hidden_state(len(predator_agents), HIDDEN_DIM)
    actor_rnn_mask = Utils.init_rnn_mask(len(predator_agents), 1)
    buffer.actor_rnn_masks[0].copy_(actor_rnn_mask)

    # There's only one critic with its own hidden state
    critic_hidden_states = Utils.init_hidden_state(1, HIDDEN_DIM)
    critic_rnn_mask = Utils.init_rnn_mask(1 , 1)
    buffer.critic_rnn_masks[0].copy_(critic_rnn_mask)

    steps = 0
    capture_step = None

    for step in range(MAX_STEPS):
        # The parallel environment updates env.agents after each step.
        if not env.agents:
            break

        #unwrapping world for reward shaping
        world = env.unwrapped.world

        # Collect observations from the agents
        active_predators = [agent for agent in predator_agents if agent in env.agents ]
        predator_obs = np.stack([observations[agent] for agent in active_predators])
        predator_obs = torch.from_numpy(predator_obs).float().to(device)

        # now we connect the policy
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        # Now the critic does its stuff
        global_state = env.state()
        global_state = torch.as_tensor(global_state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predator_actions, log_proba, next_actor_hidden = actor.act(predator_obs, actor_hidden_states, actor_rnn_mask)
            value, next_critic_hidden = critic(global_state, critic_hidden_states, critic_rnn_mask)

        # Replace the current random action for one action from the policy
        for i, agent in enumerate(active_predators):
           actions[agent] = int(predator_actions[i].item())

        # Execute the actions
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        if capture_step is None and all(terminations.get(agent, False) for agent in prey_agents):
            capture_step = step + 1

        #compute shaping rewards
        targets = [a for a in world.agents if not a.adversary]
        adversaries = [a for a in world.agents if a.adversary]
        shape_rewards = 0
        for adv in adversaries:
            shape_rewards -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))for a in targets])

        if SHAPING:
            team_rewards = torch.tensor(
                        [
                            rewards[agent] + shape_rewards
                            for agent in predator_agents
                        ],
                        dtype=torch.float32,
                        device=device,
                    )

        # compute the rewards
        else:
            team_rewards = torch.tensor(
                [
                    rewards[agent]
                    for agent in predator_agents
                ],
                dtype=torch.float32,
                device=device,
            )

        team_reward = team_rewards.mean()
        episode_reward += team_reward.item()

        # Flag the mask so it preserves or halt the status
        episode_done = all(terminations.get(agent, False) or truncations.get(agent, False) for agent in predator_agents)

        if episode_done:
            mask_value = 0.0
            next_actor_obs = torch.zeros_like(predator_obs)
            next_state = torch.zeros_like(torch.tensor(env.state(), dtype=torch.float32, device=device))
        else:
            mask_value = 1.0
            next_actor_obs = torch.stack([
                torch.as_tensor(
                    next_observations[agent],
                    dtype=torch.float32,
                    device=device,
                ) for agent in predator_agents
            ])
            next_state = torch.as_tensor(
                env.state(),
                dtype=torch.float32,
                device=device,
            )

        next_actor_rnn_mask = torch.full((len(predator_agents), 1), mask_value, dtype=torch.float32).to(device)
        next_critic_rnn_mask = torch.full((1, 1), mask_value, dtype=torch.float32).to(device)

        buffer.insert(predator_obs,
                      # remove the first dimension so it accommodates the format of the buffer
                      global_state.squeeze(0),
                      actor_hidden_states,
                      critic_hidden_states,
                      predator_actions,
                      log_proba,
                      # same here, we have a tensor [] and expect a scalar
                      value.squeeze(),
                      team_reward,
                      next_actor_obs,
                      next_state,
                      next_actor_hidden,
                      next_critic_hidden,
                      next_actor_rnn_mask,
                      next_critic_rnn_mask)

        # Update the status
        actor_hidden_states = next_actor_hidden
        actor_rnn_mask = next_actor_rnn_mask
        critic_hidden_states = next_critic_hidden
        critic_rnn_mask = next_critic_rnn_mask
        observations = next_observations
        steps += 1

        if RENDER:
            env.render()

        if episode_done:
            break

    with torch.no_grad():
        final_state = torch.as_tensor(env.state(), dtype=torch.float32).unsqueeze(0).to(device)
        value, _ = critic(final_state, critic_hidden_states, critic_rnn_mask)
        buffer.compute_return_and_advantages(value)

    t = buffer.step

    return {
        "steps": steps,
        "rewards": episode_reward,
        "captured": capture_step is not None,
        "capture_step": capture_step,
    }





def train(seed: int, device: torch.device) -> dict[str, dict[str, float]]:
    """Train once and save the best evaluated actor."""
    set_seed(seed, device)
    env = get_env(render_mode="human" if RENDER else None)
    eval_seed = EVAL_SEED

    try:
        sample_predator_id = "adversary_0"
        obs_dim = env.observation_space(sample_predator_id).shape[0]
        action_dim = env.action_space(sample_predator_id).n
        state_dim = env.state_space.shape[0]
        predator_count = sum("adversary" in agent for agent in env.possible_agents)

        actor = Actor(obs_dim, HIDDEN_DIM, action_dim).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=5e-4, eps=1e-5)
        critic = Critic(state_dim, HIDDEN_DIM).to(device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=5e-4, eps=1e-5)
        trainer = MAPPOTrainer(
            actor, actor_optimizer, critic, critic_optimizer, ppo_epochs=5
        )

        best_eval_mean = float("-inf") #changed mean to -inf to make sure it actually saves 
        first_capture_episode = None
        recent_captures = deque(maxlen=100)
        episode_index = 0
        update_index = 0
        next_evaluation_episode = EVAL_INTERVAL

        while episode_index < NUM_EPISODES:
            rollout_buffers = []
            batch_captures = 0
            episodes_in_update = min(
                ROLLOUT_EPISODES,
                NUM_EPISODES - episode_index,
            )

            # Keep the policy fixed while collecting a larger on-policy batch.
            for _ in range(episodes_in_update):
                buffer = RolloutBuffer(
                    MAX_STEPS,
                    predator_count,
                    obs_dim,
                    state_dim,
                    HIDDEN_DIM,
                    device,
                )
                episode_seed = seed + episode_index
                result = run_episode(
                    env,
                    episode_seed,
                    actor,
                    critic,
                    buffer,
                    device,
                )
                rollout_buffers.append(buffer)

                episode_index += 1
                recent_captures.append(result["captured"])
                if result["captured"]:
                    batch_captures += 1
                    if first_capture_episode is None:
                        first_capture_episode = episode_index

                print(
                    f"Seed {seed} | Episode {episode_index} | "
                    f"Steps: {result['steps']} | Train return: {result['rewards']:.2f} | "
                    f"Captured: {'yes at step ' + str(result['capture_step']) if result['captured'] else 'no'} | "
                    f"Recent capture rate (last {len(recent_captures)}): "
                    f"{np.mean(recent_captures):.0%} | "
                    f"First capture episode: {first_capture_episode or 'none'}"
                )

            print(
                f"Seed {seed} | Rollout batch capture rate: "
                f"{batch_captures / episodes_in_update:.0%} "
                f"({batch_captures}/{episodes_in_update})"
            )

            # Perform one PPO update using every episode collected above.
            trainer.update(rollout_buffers)
            update_index += 1

            should_evaluate = (
                episode_index >= next_evaluation_episode
                or episode_index == NUM_EPISODES
            )
            if should_evaluate:
                while next_evaluation_episode <= episode_index:
                    next_evaluation_episode += EVAL_INTERVAL

                eval_episodes = actor.evaluate_policy(device, eval_seed, int(os.getenv("EVAL_EPISODES", np.nan)))
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
                                "num_episodes": NUM_EPISODES,
                                "rollout_episodes": ROLLOUT_EPISODES,
                                "max_steps": MAX_STEPS,
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
    """Compare a trained actor and random policy on identical held-out seeds."""
    print(
        f"Testing on {TEST_EPISODES} held-out episodes "
        f"(seeds {TEST_SEED}–{TEST_SEED + TEST_EPISODES - 1})..."
    )
    results = {}
    for label, policy, mode, mask in ( #add masking boolean, to compare no mask + mask
        ("PPO deterministic", actor, "argmax", False),
        ("PPO sampled", actor, "sample", False),
        ("PPO deterministic obs loss", actor, "argmax",True),
        ("PPO sampled obs loss", actor, "sample", True),
        ("Random", actor, "random", False),
    ):
        episodes = actor.evaluate_policy(
            device,
            TEST_SEED,
            TEST_EPISODES,
            action_mode=mode,
            masking=mask
        )
        returns = [episode["return"] for episode in episodes]
        mean = float(np.mean(returns))
        std = float(np.std(returns))
        capture_rate = float(np.mean([episode["captured"] for episode in episodes]))
        capture_steps = [episode["capture_step"] for episode in episodes if episode["captured"]]
        mean_capture_step = float(np.mean(capture_steps)) if capture_steps else float("nan")
        results[label] = mean
        print(
            f"{label}: return {mean:.2f} +/- {std:.2f} | "
            f"captures {capture_rate:.0%} | "
            f"mean capture step {mean_capture_step:.1f}"
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

    print(f"Running {NUM_EPISODES} MAPPO episodes with seed {GLOBAL_SEED}...")
    train(GLOBAL_SEED, device)


if __name__ == "__main__":
    main()
