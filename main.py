
from __future__ import annotations

import random

import numpy as np
import torch
from mpe2 import simple_tag_v3

from policies.Actor import Actor
from policies.Critic import Critic
from policies.MAPPOTrainer import MAPPOTrainer
from utils import Utils
from utils.RolloutBuffer import RolloutBuffer

# Configuration
GLOBAL_SEED = 42
NUM_EPISODES = 10
MAX_STEPS = 30
RENDER = True
HIDDEN_DIM = 64
KILLED_OBS = np.random.randint(0,3) #this one is killed

OBS_KILL_DURATION = 5
OBS_KILL_START = np.random.randint(0,MAX_STEPS-OBS_KILL_DURATION)


def get_env(render_mode: str | None = None):
    """Create the cooperative pursuit environment."""
    return simple_tag_v3.parallel_env(
        render_mode=render_mode,
        max_cycles=MAX_STEPS,
        terminate_on_success=True,
    )

def set_seed(seed: int) -> None:
    """Seed the main random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_episode(env,
                episode: int,
                actor: Actor,
                critic: Critic,
                buffer: RolloutBuffer,
                device: torch.device) -> dict:
    """Run one episode using a random policy."""
    # we can make GLOBAL_SEED + episode if we would like a different env per episode
    observations, infos = env.reset(seed=GLOBAL_SEED + episode)
    print(observations)
    team_reward = 0.0
    predator_agents = []

    # Seed each action space for reproducible random actions.
    for agent in env.possible_agents:
        env.action_space(agent).seed(GLOBAL_SEED)
        if "adversary" in agent:
            predator_agents.append(agent)

    # The batch size corresponds to the number of agents to train
    actor_hidden_states = Utils.init_hidden_state(len(predator_agents), HIDDEN_DIM)
    actor_rnn_mask = Utils.init_rnn_mask(len(predator_agents), 1)
    buffer.actor_rnn_masks[0].copy_(actor_rnn_mask)

    # There's only one critic with its own hidden state
    critic_hidden_states = Utils.init_hidden_state(1, HIDDEN_DIM)
    critic_rnn_mask = Utils.init_rnn_mask(1 , 1)
    buffer.critic_rnn_masks[0].copy_(critic_rnn_mask)

    steps = 0

    for step in range(MAX_STEPS):
        # The parallel environment updates env.agents after each step.
        if not env.agents:
            break

        # Collect observations from the agents
        active_predators = [agent for agent in predator_agents if agent in env.agents ]
        predator_obs = np.stack([observations[agent] for agent in active_predators])
        #kill one
        masked_predator_obs = predator_obs.copy()
        obs_masked_step_max = OBS_KILL_START + OBS_KILL_DURATION
        if step >= OBS_KILL_START and step < obs_masked_step_max:
            masked_predator_obs[KILLED_OBS][:] = 0
        print(masked_predator_obs)
        masked_predator_obs = torch.from_numpy(masked_predator_obs).float().to(device)

        # now we connect the policy
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        # Now the critic does its stuff
        global_state = env.state()
        global_state = torch.as_tensor(global_state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predator_actions, log_proba, next_actor_hidden = actor.act(masked_predator_obs, actor_hidden_states, actor_rnn_mask)
            value, next_critic_hidden = critic(global_state, critic_hidden_states, critic_rnn_mask)

        # Replace the current random action for one action from the policy
        for i, agent in enumerate(active_predators):
            actions[agent] = int(predator_actions[i].item())

        # Execute the actions
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # compute the rewards
        team_rewards = torch.tensor(
            [
                rewards[agent]
                for agent in predator_agents
            ],
            dtype=torch.float32,
            device=device,
        )

        team_reward = team_rewards.mean()

        # Flag the mask so it preserves or halt the status
        episode_done = all(terminations.get(agent, False) or truncations.get(agent, False) for agent in predator_agents)

        if episode_done:
            mask_value = 0.0
            next_actor_obs = torch.zeros_like(masked_predator_obs)
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

        buffer.insert(masked_predator_obs,
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
    """
    print("\n--- BUFFER DEBUG ---")
    print("steps:", t)
    print("rewards:   ", buffer.rewards[:t])
    print("values:    ", buffer.values[:t + 1])
    print("advantages:", buffer.advantages[:t])
    print("returns:   ", buffer.returns[:t])
    """
    return {
        "episode": episode + 1,
        "steps": steps,
        "rewards": team_reward,
    }

def main() -> None:
    set_seed(GLOBAL_SEED)
    device = Utils.get_torch_device()
    print(f"PyTorch device: {device}")
    print(f"Running {NUM_EPISODES} episodes...")

    env = get_env(render_mode="human" if RENDER else None)

    try:
        # Policy shapes initialization
        sample_predator_id = "adversary_0"
        obs_dim = env.observation_space(sample_predator_id).shape[0]
        action_dim = env.action_space(sample_predator_id).n
        state_dim = env.state_space.shape[0]
        total_agents = 0
        for agent in env.possible_agents:
            total_agents += 1 if "adversary" in agent else 0

        # Now create the shared actor (It is a single policy shared between all the different agents) + one critic for the whole env
        actor = Actor(obs_dim, HIDDEN_DIM, action_dim).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=5e-4, eps=1e-5)
        critic = Critic(state_dim, HIDDEN_DIM).to(device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=5e-4, eps=1e-5)

        leTrainer = MAPPOTrainer(actor, actor_optimizer, critic, critic_optimizer)

        for episode in range(NUM_EPISODES):
            buffer = RolloutBuffer(
                MAX_STEPS,
                total_agents,
                obs_dim,
                state_dim,
                HIDDEN_DIM,
                device
            )

            result = run_episode(env, episode, actor, critic, buffer, device)
            training_info = leTrainer.update(buffer)

            print(
                f"Episode {result['episode']:02d} | "
                f"Steps: {result['steps']:02d} | "
                f"Total reward: {result["rewards"]:.2f} | "
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()
