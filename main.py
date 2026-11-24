"""MAPPO pursuit experiments.

  python main.py train [--pursuers N] [--prey random|fleeing] [--seed S] [--masked]
      Variation 1 (clean) or Variation 2 (--masked) training. Writes runs/<run>/ with
      best_actor.pt, best_noloss_actor.pt, final_actor.pt, learning_curve.csv, config.json.

  python main.py sweep --checkpoint runs/<run>/best_actor.pt [...]
  python main.py sweep --random --pursuers N --prey random|fleeing
      Variation 3 evaluation sweep. One CSV row per episode in runs/sweep/.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from policies.Actor import Actor
from policies.Critic import Critic
from policies.MAPPOTrainer import MAPPOTrainer
from utils import Utils
from utils.Blackout import (FILL_METHODS, BlackoutApplier, actor_obs_dim, critic_state_dim,
                            paired_draw, random_training_blackout, validation_blackout)
from utils.Config import CONFIG
from utils.EnvUtils import (get_env, seed_action_spaces, set_seed, split_agents,
                            training_episode_seed)
from utils.Evaluation import evaluate, run_eval_episode
from utils.Prey import prey_actions, prey_rng
from utils.RolloutBuffer import RolloutBuffer

LEARNING_RATE = 5e-4
SWEEP_LENGTHS = (5, 10, 20, 40, None)  # None = permanent


# --------------------------------------------------------------------------- naming

def prey_label() -> str:
    if CONFIG.prey_type == "fleeing":
        return f"fleeing-p{CONFIG.flee_prob:g}"
    return "random"


def variant_label() -> str:
    return "masked" if CONFIG.masked_training else "clean"


def run_name() -> str:
    return f"{variant_label()}_n{CONFIG.num_pursuers}_{prey_label()}_s{CONFIG.seed}"


# --------------------------------------------------------------------------- training

def critic_state(env, applier: BlackoutApplier, step: int, num_pursuers: int) -> np.ndarray:
    """Global state for the centralized critic; with critic_blind_mask, append who is blind."""
    state = np.asarray(env.state(), dtype=np.float32)
    if CONFIG.critic_blind_mask:
        state = np.concatenate([state, applier.blind_mask(step, num_pursuers)])
    return state


def run_episode(env, episode_seed: int, actor: Actor, critic: Critic,
                buffer: RolloutBuffer, device: torch.device) -> dict:
    """Collect one training episode into ``buffer``."""
    observations, _ = env.reset(seed=episode_seed)
    seed_action_spaces(env, episode_seed)
    predator_agents, prey_agents = split_agents(env)
    rng = prey_rng(episode_seed)
    blackout = (random_training_blackout(episode_seed, len(predator_agents))
                if CONFIG.masked_training else None)
    applier = BlackoutApplier(blackout)

    actor_hidden_states = Utils.init_hidden_state(len(predator_agents), CONFIG.hidden_dim)
    actor_rnn_mask = Utils.init_rnn_mask(len(predator_agents), 1)
    buffer.actor_rnn_masks[0].copy_(actor_rnn_mask)
    critic_hidden_states = Utils.init_hidden_state(1, CONFIG.hidden_dim)
    critic_rnn_mask = Utils.init_rnn_mask(1, 1)
    buffer.critic_rnn_masks[0].copy_(critic_rnn_mask)

    episode_reward = 0.0
    steps = 0
    capture_step = None
    world = env.unwrapped.world

    for step in range(CONFIG.max_steps):
        if not env.agents:
            break

        raw_obs = np.stack([observations[agent] for agent in predator_agents])
        # Actors only see the (possibly blinded) observation; the buffer stores exactly
        # what the actor acted on so the PPO ratio is computed on the same inputs.
        seen_obs = applier.actor_inputs(raw_obs, step)
        predator_obs = torch.from_numpy(seen_obs).float().to(device)

        # The critic always gets the full state, rebuilt from the world (plus who is blind, if enabled).
        global_state = torch.as_tensor(critic_state(env, applier, step, len(predator_agents)),
                                       dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predator_actions, log_proba, next_actor_hidden = actor.act(
                predator_obs, actor_hidden_states, actor_rnn_mask)
            value, next_critic_hidden = critic(global_state, critic_hidden_states, critic_rnn_mask)

        actions = prey_actions(env, prey_agents, rng)
        for i, agent in enumerate(predator_agents):
            actions[agent] = int(predator_actions[i].item())

        next_observations, rewards, terminations, truncations, _ = env.step(actions)
        if capture_step is None and all(terminations.get(agent, False) for agent in prey_agents):
            capture_step = step + 1

        shape_rewards = 0.0
        if CONFIG.shaping:
            targets = [a for a in world.agents if not a.adversary]
            for adv in (a for a in world.agents if a.adversary):
                shape_rewards -= 0.1 * min(
                    float(np.sqrt(np.sum(np.square(t.state.p_pos - adv.state.p_pos)))) for t in targets)
        team_reward = torch.tensor(
            [rewards[agent] + shape_rewards for agent in predator_agents],
            dtype=torch.float32, device=device).mean()
        episode_reward += team_reward.item()

        episode_done = all(terminations.get(a, False) or truncations.get(a, False)
                           for a in predator_agents)
        mask_value = 0.0 if episode_done else 1.0
        next_state = torch.as_tensor(critic_state(env, applier, step + 1, len(predator_agents)),
                                     dtype=torch.float32, device=device)
        if episode_done:
            next_state = torch.zeros_like(next_state)
        next_actor_rnn_mask = torch.full((len(predator_agents), 1), mask_value,
                                         dtype=torch.float32, device=device)
        next_critic_rnn_mask = torch.full((1, 1), mask_value, dtype=torch.float32, device=device)

        buffer.insert(predator_obs,
                      global_state.squeeze(0),
                      actor_hidden_states,
                      critic_hidden_states,
                      predator_actions,
                      log_proba,
                      value.squeeze(),
                      team_reward,
                      # next observations are not used by the update; stored for completeness
                      torch.zeros_like(predator_obs),
                      next_state,
                      next_actor_hidden,
                      next_critic_hidden,
                      next_actor_rnn_mask,
                      next_critic_rnn_mask)

        actor_hidden_states = next_actor_hidden
        actor_rnn_mask = next_actor_rnn_mask
        critic_hidden_states = next_critic_hidden
        critic_rnn_mask = next_critic_rnn_mask
        observations = next_observations
        steps += 1

        if CONFIG.render:
            env.render()
        if episode_done:
            break

    with torch.no_grad():
        final_state = torch.as_tensor(critic_state(env, applier, steps, len(predator_agents)),
                                      dtype=torch.float32).unsqueeze(0).to(device)
        value, _ = critic(final_state, critic_hidden_states, critic_rnn_mask)
        buffer.compute_return_and_advantages(value)

    return {
        "steps": steps,
        "rewards": episode_reward,
        "captured": capture_step is not None,
        "capture_step": capture_step,
    }


def validate(env, actor: Actor, device: torch.device) -> dict:
    """Validation seeds only: no loss, and the fixed hard blackout (N-1 blinded)."""
    n = CONFIG.num_pursuers
    no_loss = evaluate(env, actor, device, CONFIG.eval_seed, CONFIG.eval_episodes, lambda s: None)
    blackout = evaluate(env, actor, device, CONFIG.eval_seed, CONFIG.eval_episodes,
                        lambda s: validation_blackout(s, n))
    return {
        "val_capture_noloss": no_loss["capture_rate"],
        "val_capture_blackout": blackout["capture_rate"],
        "val_return_noloss": no_loss["env_return"],
        "val_return_blackout": blackout["env_return"],
        # Same selection rule for clean and masked: mean of the two capture rates,
        # ties broken by mean env return.
        "selection_score": (no_loss["capture_rate"] + blackout["capture_rate"]) / 2,
        "selection_tiebreak": (no_loss["env_return"] + blackout["env_return"]) / 2,
    }


def save_checkpoint(path: Path, actor: Actor, obs_dim: int, action_dim: int,
                    selection: str, progress: dict, validation: dict, trainer: MAPPOTrainer) -> None:
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "variant": variant_label(),
        "selection": selection,
        "training_seed": CONFIG.seed,
        **progress,
        "validation": validation,
        "config": CONFIG.as_dict(),
        "trainer": {
            "ppo_epochs": trainer.ppo_epochs,
            "clip_param": trainer.clip_param,
            "entropy_coef": trainer.entropy_coef,
            "recurrent_chunk_length": trainer.recurrent_chunk_length,
            "learning_rate": LEARNING_RATE,
            "gamma": 0.99,
            "gae_lambda": 0.95,
        },
    }, path)


def train(device: torch.device) -> Path:
    seed = CONFIG.seed
    set_seed(seed, device)
    run_dir = Path(CONFIG.runs_dir) / run_name()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(CONFIG.as_dict(), indent=2))
    env = get_env(render_mode="human" if CONFIG.render else None)
    eval_env = get_env()

    try:
        n = CONFIG.num_pursuers
        obs_dim = actor_obs_dim(int(env.observation_space("adversary_0").shape[0]))
        action_dim = int(env.action_space("adversary_0").n)
        state_dim = critic_state_dim(env.state_space.shape[0], n)

        actor = Actor(obs_dim, CONFIG.hidden_dim, action_dim).to(device)
        critic = Critic(state_dim, CONFIG.hidden_dim).to(device)
        trainer = MAPPOTrainer(
            actor, torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE, eps=1e-5),
            critic, torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE, eps=1e-5),
            ppo_epochs=10,
        )

        best = {"score": (-1.0, float("-inf"))}
        best_noloss = {"score": (-1.0, float("-inf"))}
        recent_captures = deque(maxlen=100)
        episode_index = update_index = env_steps = 0
        start_time = time.perf_counter()

        curve_path = run_dir / "learning_curve.csv"
        curve_fields = ["update", "episode", "env_steps", "wall_time_s", "train_capture_recent100",
                        "val_capture_noloss", "val_capture_blackout", "val_return_noloss",
                        "val_return_blackout", "selection_score", "target_episodes"]
        target = CONFIG.num_episodes
        cap = max(CONFIG.max_episodes, CONFIG.num_episodes)
        with curve_path.open("w", newline="") as curve_file:
            curve = csv.DictWriter(curve_file, fieldnames=curve_fields)
            curve.writeheader()

            while episode_index < target:
                buffers = []
                batch_size = min(CONFIG.rollout_episodes, target - episode_index)
                batch_captures = 0
                for _ in range(batch_size):
                    buffer = RolloutBuffer(CONFIG.max_steps, n, obs_dim, state_dim,
                                           CONFIG.hidden_dim, device)
                    result = run_episode(env, training_episode_seed(seed, episode_index),
                                         actor, critic, buffer, device)
                    buffers.append(buffer)
                    episode_index += 1
                    env_steps += result["steps"]
                    recent_captures.append(result["captured"])
                    batch_captures += result["captured"]

                trainer.update(buffers)
                update_index += 1
                print(f"[{run_name()}] update {update_index} | episode {episode_index} | "
                      f"steps {env_steps} | batch capture {batch_captures}/{batch_size} | "
                      f"recent100 {np.mean(recent_captures):.0%}", flush=True)

                at_target = episode_index >= target
                if update_index % CONFIG.eval_interval_updates and not at_target:
                    continue

                validation = validate(eval_env, actor, device)
                progress = {"update": update_index, "episode": episode_index, "env_steps": env_steps}
                curve.writerow({
                    **progress,
                    "wall_time_s": round(time.perf_counter() - start_time, 2),
                    "train_capture_recent100": float(np.mean(recent_captures)),
                    **{k: validation[k] for k in curve_fields if k in validation},
                    "target_episodes": target,
                })
                curve_file.flush()
                print(f"[{run_name()}] VALIDATION update {update_index} | "
                      f"no-loss capture {validation['val_capture_noloss']:.0%} | "
                      f"blackout capture {validation['val_capture_blackout']:.0%} | "
                      f"score {validation['selection_score']:.3f}", flush=True)

                score = (validation["selection_score"], validation["selection_tiebreak"])
                if score > best["score"]:
                    best = {"score": score}
                    save_checkpoint(run_dir / "best_actor.pt", actor, obs_dim, action_dim,
                                    "best", progress, validation, trainer)
                noloss_score = (validation["val_capture_noloss"], validation["val_return_noloss"])
                if noloss_score > best_noloss["score"]:
                    best_noloss = {"score": noloss_score}
                    save_checkpoint(run_dir / "best_noloss_actor.pt", actor, obs_dim, action_dim,
                                    "best_noloss", progress, validation, trainer)
                if at_target:
                    best_capture = best_noloss["score"][0]
                    if CONFIG.val_threshold > 0 and best_capture < CONFIG.val_threshold and target < cap:
                        target = min(target + CONFIG.extension_episodes, cap)
                        print(f"[{run_name()}] best no-loss validation {best_capture:.0%} < threshold "
                              f"{CONFIG.val_threshold:.0%}: extending to {target} episodes", flush=True)
                        continue
                    save_checkpoint(run_dir / "final_actor.pt", actor, obs_dim, action_dim,
                                    "final", progress, validation, trainer)

        best_capture = best_noloss["score"][0]
        summary = {
            "episodes_used": episode_index,
            "base_episodes": CONFIG.num_episodes,
            "extended": episode_index > CONFIG.num_episodes,
            "best_noloss_validation": best_capture,
            "val_threshold": CONFIG.val_threshold,
            "reached_threshold": CONFIG.val_threshold <= 0 or best_capture >= CONFIG.val_threshold,
            "wall_time_s": round(time.perf_counter() - start_time, 1),
        }
        (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{run_name()}] done in {time.perf_counter() - start_time:.0f}s -> {run_dir} | {summary}")
        return run_dir
    finally:
        env.close()
        eval_env.close()


# --------------------------------------------------------------------------- sweep

SWEEP_FIELDS = ["method", "selection", "training_seed", "pursuers", "prey", "blackout_length",
                "num_blinded", "fill", "start_step", "test_seed", "captured", "capture_step",
                "captured_before_blackout", "env_return", "mean_distance", "spread",
                "max_angular_gap", "collisions", "steps"]


def sweep_conditions(n: int):
    yield None, 0, "none"
    for length in SWEEP_LENGTHS:
        for num_blinded in range(1, n):
            for fill in FILL_METHODS:
                yield length, num_blinded, fill


def run_sweep(actor: Actor | None, labels: dict, out_path: Path, device: torch.device) -> None:
    n = CONFIG.num_pursuers
    policy = "random" if actor is None else "argmax"
    if actor is not None:
        actor.eval()
    env = get_env()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = range(CONFIG.test_seed, CONFIG.test_seed + CONFIG.test_episodes)
    draws = {seed: paired_draw(seed, n) for seed in seeds}
    started = time.perf_counter()
    try:
        with out_path.open("w", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=SWEEP_FIELDS)
            writer.writeheader()
            for length, num_blinded, fill in sweep_conditions(n):
                captures = 0
                for seed in seeds:
                    spec = draws[seed].spec(length, num_blinded, fill) if num_blinded else None
                    result = run_eval_episode(env, actor, device, seed, spec, policy)
                    captures += result["captured"]
                    writer.writerow({
                        **labels,
                        "pursuers": n,
                        "prey": prey_label(),
                        "blackout_length": "none" if spec is None else spec.length_label,
                        "num_blinded": num_blinded,
                        "fill": fill,
                        "start_step": "" if spec is None else spec.start,
                        "test_seed": seed,
                        **{k: result[k] for k in SWEEP_FIELDS if k in result},
                    })
                out_file.flush()
                length_label = "none" if not num_blinded else ("permanent" if length is None else length)
                print(f"[{out_path.stem}] length={length_label} "
                      f"blinded={num_blinded} fill={fill}: capture {captures / len(seeds):.0%} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    finally:
        env.close()


def sweep_checkpoint(path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    saved = checkpoint["config"]
    # The scenario must match training; the test protocol comes from the current CONFIG.
    CONFIG.update(**{k: saved[k] for k in ("num_pursuers", "prey_type", "flee_prob", "flee_boundary",
                                           "flee_push", "max_steps", "hidden_dim", "masked_training",
                                           "seed")})
    # Checkpoints from before the redo lack these keys -> original blackout behaviour.
    CONFIG.update(**{k: bool(saved.get(k, False))
                     for k in ("keep_proprioception", "blind_flag", "critic_blind_mask")})
    actor = Actor(checkpoint["obs_dim"], CONFIG.hidden_dim, checkpoint["action_dim"]).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    labels = {"method": checkpoint["variant"], "selection": checkpoint["selection"],
              "training_seed": checkpoint["training_seed"]}
    out = Path(CONFIG.runs_dir) / "sweep" / f"{path.parent.name}__{checkpoint['selection']}.csv"
    run_sweep(actor, labels, out, device)


def sweep_random(device: torch.device) -> None:
    labels = {"method": "random", "selection": "", "training_seed": ""}
    out = Path(CONFIG.runs_dir) / "sweep" / f"random_n{CONFIG.num_pursuers}_{prey_label()}.csv"
    run_sweep(None, labels, out, device)


# --------------------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_scenario(p):
        p.add_argument("--pursuers", type=int, dest="num_pursuers")
        p.add_argument("--prey", choices=["random", "fleeing"], dest="prey_type")
        p.add_argument("--flee-prob", type=float, dest="flee_prob")
        p.add_argument("--episodes", type=int, help="training episodes (train) or test episodes per condition (sweep)")
        p.add_argument("--runs-dir", dest="runs_dir")

    train_parser = sub.add_parser("train", help="Variation 1 (clean) or 2 (--masked) training")
    add_scenario(train_parser)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--masked", action="store_true", dest="masked_training", default=None)
    train_parser.add_argument("--eval-interval", type=int, dest="eval_interval_updates")
    train_parser.add_argument("--eval-episodes", type=int, dest="eval_episodes")

    sweep_parser = sub.add_parser("sweep", help="Variation 3 evaluation sweep")
    add_scenario(sweep_parser)
    sweep_parser.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    sweep_parser.add_argument("--random", action="store_true", help="random-policy baseline")

    args = vars(parser.parse_args())
    command = args.pop("command")
    episodes = args.pop("episodes")
    checkpoints = args.pop("checkpoint", [])
    random_baseline = args.pop("random", False)
    if episodes is not None:
        args["num_episodes" if command == "train" else "test_episodes"] = episodes
    CONFIG.update(**args)

    device = Utils.get_torch_device()
    print(f"PyTorch device: {device}")
    if command == "train":
        train(device)
        return
    if not checkpoints and not random_baseline:
        parser.error("sweep needs --checkpoint PATH [...] and/or --random")
    for path in checkpoints:
        sweep_checkpoint(path, device)
    if random_baseline:
        sweep_random(device)


if __name__ == "__main__":
    main()
