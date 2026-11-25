from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from policies.CorrectionBlock import CorrectionBlock
from utils.RolloutBuffer import RolloutBuffer

class CorrectionTrainer:
    """Train a CorrectionBlock from complete observations and prior estimates."""

    def __init__(
        self,
        corrector: CorrectionBlock,
        optimizer: Optimizer,
        correction_epochs: int = 1,
        max_grad_norm: float | None = 10.0,
    ) -> None:
        if correction_epochs <= 0:
            raise ValueError("correction_epochs must be positive.")

        self.corrector = corrector
        self.optimizer = optimizer
        self.correction_epochs = correction_epochs
        self.max_grad_norm = max_grad_norm

    @staticmethod
    def _collect_observations(buffers: list[RolloutBuffer]) -> torch.Tensor:
        """Collect valid ground-truth observations into a [B, N, D] batch."""
        return torch.cat(
            [
                buffer.observations[:buffer.step]
                for buffer in buffers
            ],
            dim=0,
        ).detach()

    @staticmethod
    def _sample_lost_mask(
        num_agents: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Mark every agent except one randomly selected reliable agent as lost."""
        if num_agents < 2:
            raise ValueError("Correction training requires at least two agents.")

        reliable_agent = torch.randint(
            0,
            num_agents,
            (1,),
            device=device,
        ).item()
        return torch.tensor(
            [index != reliable_agent for index in range(num_agents)],
            dtype=torch.bool,
            device=device,
        )

    def update(
        self,
        buffers: list[RolloutBuffer],
        prior_observations: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Perform correction updates and return scalar training metrics.

        If no prior observations are supplied, complete observations are used as
        temporary oracle priors. World-model priors must have shape [B, N, D].
        """
        if not buffers:
            raise ValueError("At least one rollout buffer is required.")

        true_observations = self._collect_observations(buffers)
        if true_observations.ndim != 3:
            raise ValueError(
                "Complete observations must have shape [B, N, D], got "
                f"{tuple(true_observations.shape)}."
            )

        batch_size, num_agents, _ = true_observations.shape
        if batch_size == 0:
            raise ValueError("The rollout buffers contain no completed steps.")

        if prior_observations is not None:
            if prior_observations.shape != true_observations.shape:
                raise ValueError(
                    "prior_observations must match the complete observations: "
                    f"expected {tuple(true_observations.shape)}, got "
                    f"{tuple(prior_observations.shape)}."
                )
            prior_observations = prior_observations.to(
                device=true_observations.device,
                dtype=true_observations.dtype,
            )

        full_prior = (
            true_observations.detach().clone()
            if prior_observations is None
            else prior_observations.detach().clone()
        )

        self.corrector.train()
        epoch_losses: list[float] = []

        for _ in range(self.correction_epochs):
            lost_mask = self._sample_lost_mask(
                num_agents,
                true_observations.device,
            )

            real_input = true_observations.clone()
            real_input[:, lost_mask] = 0.0
            prior_input = full_prior.clone()
            prior_input[:, ~lost_mask] = 0.0
            if self.corrector.use_trust:
                error = F.mse_loss(real_input+prior_input, true_observations, reduction='none')
                agent_error = error.mean(dim=-1)
                trust_score = 1/(1+agent_error)

            real_flat = real_input.reshape(batch_size, -1)
            prior_flat = prior_input.reshape(batch_size, -1)

            self.optimizer.zero_grad()
            if self.corrector.use_trust:
                predicted_observations = self.corrector(
                    real_flat,
                    prior_flat,
                    trust_score,
                ).reshape_as(true_observations)
            else:
                predicted_observations = self.corrector(
                    real_flat,
                    prior_flat,
                ).reshape_as(true_observations)
            loss = F.mse_loss(
                predicted_observations[:, lost_mask],
                true_observations[:, lost_mask],
            )

            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.corrector.parameters(),
                    self.max_grad_norm,
                )
            self.optimizer.step()
            epoch_losses.append(loss.item())

        return {
            "correction_loss": sum(epoch_losses) / len(epoch_losses),
        }
