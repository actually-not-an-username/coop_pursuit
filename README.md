# coop_pursuit

PyTorch implementation of **RMIO: A Model-Based MARL Framework for Scenarios
with Observation Loss in Some Agents** (arXiv:2411.19639).

## Included algorithms

- Multi-agent recurrent state-space world model (RSSM), with cross-agent attention,
  Gaussian latent priors/posteriors, and observation, reward, and continuation heads.
- Attention-based correction of observations missing for individual agents.
- Shared GRU actor with centralized critic, MAPPO clipped policy loss, and GAE.
- Gaussian reward smoothing and separate real and imagined episode replay buffers.
- Alternating training for the world model, policy-conditioned imagination, MAPPO,
  and observation correction.
- Inference that uses decentralized policy execution during full observations and
  synchronizes the world model from recent history when observation loss occurs.

## Setup

This project uses Python 3.14 and `uv`:

```bash
uv sync
```

Device selection is automatic: MPS on macOS, CUDA on Linux when available, and
CPU otherwise. Pass `device="cpu"` (or another PyTorch device) to `RMIOTrainer`
or `RMIOInference` to override the selection.

## Models and trainer

```python
from rmio import (
    MultiAgentWorldModel, ObservationCorrector, RecurrentActorCritic,
    RMIOConfig, RMIOInference, RMIOTrainer, Transition,
)

agents, observation_dim, discrete_actions = 3, 12, 5
world_model = MultiAgentWorldModel(observation_dim, discrete_actions)
corrector = ObservationCorrector(observation_dim)
policy = RecurrentActorCritic(observation_dim, discrete_actions, agents)
controller = RMIOInference(world_model, corrector, policy)
trainer = RMIOTrainer(world_model, corrector, policy, RMIOConfig())
```

`controller.act(observations, previous_actions, available)` accepts observations
of shape `[batch, agents, observation_dim]`, previous discrete action indices of
shape `[batch, agents]` (pass `None` at the first step), and a boolean availability
mask `[batch, agents]`. It returns actions, action log probabilities, centralized
values, and completed observations.

The paper does not define a particular environment API. The environment adapter
should collect real episodes as lists of `Transition` objects, with per-agent
observations and next observations shaped `[agents, observation_dim]`, discrete
actions shaped `[agents]` (or continuous actions `[agents, action_dim]`), team
rewards, and a termination flag. Add episodes using `trainer.add_real_episode(...)`
and call `trainer.train_iteration()` to perform the alternating RMIO updates.
Discrete actions are one-hot encoded internally for the world model.
