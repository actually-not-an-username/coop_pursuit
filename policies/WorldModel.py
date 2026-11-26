import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import utils.Utils as Utils

import os
from dotenv import load_dotenv

# Configuration
load_dotenv("config.env")


class CommunicationBlock(nn.Module):
    def __init__(self,
                 stochastic_dim: int,
                 action_dim: int,
                 embed_dim: int,
                 num_heads: int = 4,
                 num_layers: int = 2,):
        super().__init__()

        # Encode inputs
        self.input_proj = nn.Linear(
            stochastic_dim + action_dim,
            embed_dim
        )

        fnn_multiplier = int(os.getenv("FNN_MULTIPLIER", 4))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=fnn_multiplier * embed_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

    def forward(self, z, action):
        x = torch.cat([z, action], dim=-1)
        x = self.input_proj(x)

        # Attention across agents
        return self.transformer(x)

class WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int,
        deterministic_dim: int,
        stochastic_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()

        # By default, an adversary observation is (16,) but it does scale with the number of agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim

        # Encode observations
        self.encoder =  Utils.create_mlp_layers(obs_dim, embed_dim, 2)

        # agent communication block (it shares info between agent observations)
        self.communication_block = CommunicationBlock(stochastic_dim, action_dim, embed_dim, num_heads)

        # Recurrent dynamics - This is GRUCell because it processes a single time
        # step rather than the complete sequence GRU
        self.gru = nn.GRUCell(embed_dim, deterministic_dim)

        # Prior p(z_t | h_t), posterior q(z_t | h_t, o_t).
        # Predict stochastic prior
        self.prior = Utils.create_mlp_layers(deterministic_dim, 2 * stochastic_dim, 2)
        # Infer stochastic posterior
        self.posterior = Utils.create_mlp_layers(deterministic_dim + embed_dim, 2 * stochastic_dim, 2)

        # Predict observations observations
        self.observation_head = Utils.create_mlp_layers(deterministic_dim + stochastic_dim, 2 * obs_dim, 2)

        # Predict rewards
        self.reward_head = Utils.create_mlp_layers(deterministic_dim + stochastic_dim,2, 2)

        # Predict episode continuation
        self.continue_head = Utils.create_mlp_layers(deterministic_dim + stochastic_dim,1, 2)

    def initial_state(self, batch_size, num_agents):
        h = Utils.init_rnn_mask(batch_size, self.deterministic_dim, num_agents)
        z = Utils.init_rnn_mask(batch_size, self.stochastic_dim, num_agents)
        return h, z

    @staticmethod
    def get_dist(params):
        mean, raw_std = params.chunk(2, dim=-1)
        std = F.softplus(raw_std) + 0.1
        return D.Normal(mean, std)

    def recurrent_step(self, h, z, action,):
        batch_size, num_agents, _ = z.shape

        e = self.communication_block(z, action)
        h_new = self.gru(e.reshape(batch_size*num_agents, -1), h.reshape(batch_size*num_agents, -1))

        return h_new.reshape(batch_size, num_agents, self.deterministic_dim)

    def observe_step(self, h, z, action, obs, obs_mask):
        # Advance deterministic state
        h = self.recurrent_step(h, z, action)

        # Prior: dynamics-only prediction
        prior_dist = self.get_dist(self.prior(h))

        # Posterior: observation-conditioned inference
        # Encode current observation
        obs_embed = self.encoder(obs)
        post_input = torch.cat([h, obs_embed], dim=-1)
        post_dist = self.get_dist(self.posterior(post_input))

        # since not all agents lose observations we would need z sampled from both prior and posterior
        z_prior = prior_dist.rsample()
        z_post = post_dist.rsample()

        # agents with observation -> posterior, blind agents -> prior
        z = torch.where(obs_mask.bool(), z_post, z_prior)

        return h, z, prior_dist, post_dist

    def imagine_step(self, h, z, action):
        # No observation is provided
        h = self.recurrent_step(h, z, action)

        prior_dist = self.get_dist(self.prior(h))
        z = prior_dist.rsample()

        return h, z, prior_dist

    def predict(self, h, z):
        state = torch.cat([h, z], dim=-1)

        obs_dist = self.get_dist(self.observation_head(state))
        reward_dist = self.get_dist(self.reward_head(state))
        discount_logits = self.continue_head(state)

        return obs_dist, reward_dist, discount_logits
