import random

import torch
import numpy as np
import os
from mpe2 import simple_tag_v3
from dotenv import load_dotenv

# Configuration
load_dotenv("config.env")

MAX_STEPS=int(os.getenv("MAX_STEPS", np.nan))

def set_seed(seed: int, device: torch.device) -> None:
    """Seed the main random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)

def get_env(render_mode: str | None = None):
    """Create the cooperative pursuit environment."""
    return simple_tag_v3.parallel_env(
        render_mode=render_mode,
        max_cycles=MAX_STEPS,
        terminate_on_success=True,
    )