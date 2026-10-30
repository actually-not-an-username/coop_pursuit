import torch
from torch import nn
from torch.distributions import Categorical

from policies.Rnn import RNNLayer
from utils import Utils

class Actor(nn.Module):
    """
        Shared decentralized actor.

        Each agent receives:
            - its own local observation
            - its own GRU hidden state

        It produces:
            - an action distribution
            - the next GRU hidden state

        The network parameters can be shared between al homogeneous adversaries,
        but each adversary MUST maintain its own recurrent hidden state.
        """

    def __init__(self, observation_dim, hidden_dim = 64, action_dim = 5):
        super().__init__()

        # MLP with orthogonal init as implemented in MAPPO
        self.mlp = Utils.create_mlp_layers(observation_dim, hidden_dim)
        self.rnn = RNNLayer(hidden_dim, hidden_dim)

        # keep the last gain small so the initial distribution is close to uniform, in theory given
        # n epochs this clipping shouldn't affect the final output
        self.action_head = Utils.init_orthogonal_layer(nn.Linear(hidden_dim, action_dim), gain=0.01)

    def forward(self, observation, hidden_state, mask):
        x = self.mlp(observation)
        x, next_hidden_state = self.rnn(x, hidden_state, mask)
        logits = self.action_head(x)
        dist = Categorical(logits=logits)

        return dist, next_hidden_state

    @torch.no_grad()
    def act(self, observation, hidden_state, mask):
        dist, next_hidden_state = self.forward(observation, hidden_state, mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, next_hidden_state

    def evaluate_actions(self, observation, hidden_state, actions, mask):
        dist, next_hidden_state = self.forward(observation, hidden_state, mask)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, next_hidden_state

