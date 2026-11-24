import random

import numpy as np
import torch
from mpe2 import simple_tag_v3

from utils.Config import CONFIG

# Training episode seeds are away from validation sedds (10,000+) and test seeds (20,000+)
# No seed happens twice
TRAIN_SEED_BASE = 1_000_000
TRAIN_SEED_STRIDE = 1_000_000


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def training_episode_seed(training_seed: int, episode_index: int) -> int:
    return TRAIN_SEED_BASE + training_seed * TRAIN_SEED_STRIDE + episode_index


def get_env(render_mode: str | None = None):
    """Create the cooperative pursuit environment for the current CONFIG."""
    return simple_tag_v3.parallel_env(
        render_mode=render_mode,
        num_good=1,
        num_adversaries=CONFIG.num_pursuers,
        num_obstacles=2,
        max_cycles=CONFIG.max_steps,
        terminate_on_success=True,
    )


def split_agents(env) -> tuple[list[str], list[str]]:
    predators = [agent for agent in env.possible_agents if "adversary" in agent]
    prey = [agent for agent in env.possible_agents if "adversary" not in agent]
    return predators, prey


def seed_action_spaces(env, episode_seed: int) -> None:
    for agent_index, agent in enumerate(env.possible_agents):
        env.action_space(agent).seed(episode_seed + agent_index + 1)
