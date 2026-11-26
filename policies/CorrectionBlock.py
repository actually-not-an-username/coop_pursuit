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
    def __init__(self, input_dim, feature_dim):
        super(CorrectionBlock, self).__init__()
        self.real_encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU()
        )
        self.prior_encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU()
        )
        self.corrector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU()
        )
        # self.attention = AttentionModule(feature_dim)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, input_dim)
        )
    
    def forward(self, real_observation, prior_observation):
        real_feature = self.real_encoder(real_observation)
        prior_feature = self.prior_encoder(prior_observation)
        feature = torch.cat((real_feature, prior_feature), dim=-1)
        correct_feature = self.corrector(feature)
        # correct_feature = self.attention(correct_feature)
        return self.decoder(correct_feature)
        # return correct_feature