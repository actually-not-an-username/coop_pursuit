"""
Original correction block adapted from:
Shi Zifeng et al., GAWM / RMIO implementation
https://github.com/ShiZifeng/GAWM

Reused with author's permission for ECE 780.
"""
import torch
from torch import nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_weights = F.softmax(q.matmul(k.transpose(-2, -1)) / (x.size(-1) ** 0.5), dim=-1)
        return attention_weights.matmul(v)

class CorrectionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        trust_dim: int | None = None,
    ) -> None:
        super(CorrectionBlock, self).__init__()
        self.use_trust = trust_dim is not None
        self.trust_dim = trust_dim
        self.real_encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU()
        )
        self.prior_encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU()
        )
        self.trust_encoder = (
            nn.Linear(trust_dim, feature_dim)
            if trust_dim is not None
            else None
        )
        corrector_input_dim = feature_dim * (3 if self.use_trust else 2)
        self.corrector = nn.Sequential(
            nn.Linear(corrector_input_dim, feature_dim),
            nn.ReLU()
        )
        # self.attention = AttentionModule(feature_dim)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, input_dim)
        )
    
    def forward(
        self,
        real_observation: torch.Tensor,
        prior_observation: torch.Tensor,
        trust_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        real_feature = self.real_encoder(real_observation)
        prior_feature = self.prior_encoder(prior_observation)
        features = [real_feature, prior_feature]

        if self.use_trust:
            if trust_score is None:
                raise ValueError(
                    "trust_score is required when the correction block uses trust."
                )
            if trust_score.shape[:-1] != real_observation.shape[:-1]:
                raise ValueError(
                    "trust_score and observations must have matching batch "
                    f"dimensions, got {tuple(trust_score.shape)} and "
                    f"{tuple(real_observation.shape)}."
                )
            if trust_score.shape[-1] != self.trust_dim:
                raise ValueError(
                    f"Expected {self.trust_dim} trust scores, got "
                    f"{trust_score.shape[-1]}."
                )
            if self.trust_encoder is None:
                raise RuntimeError("Trust encoder was not initialized.")
            features.append(self.trust_encoder(trust_score))
        elif trust_score is not None:
            raise ValueError(
                "trust_score was provided to a correction block without trust."
            )

        feature = torch.cat(features, dim=-1)
        correct_feature = self.corrector(feature)
        # correct_feature = self.attention(correct_feature)
        return self.decoder(correct_feature)
        # return correct_feature
