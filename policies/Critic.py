import torch
from torch import nn
from utils import Utils
from policies.Rnn import RNNLayer

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dimension = 64, output_dim = 1):
        super().__init__()

        # MLP with orthogonal init as implemented in MAPPO
        self.mlp = Utils.create_mlp_layers(state_dim, hidden_dimension)
        self.rnn = RNNLayer(hidden_dimension, hidden_dimension)

        # Unlike the actor head value head outputs a scalar estimate V(s_t).
        self.value_head = Utils.init_orthogonal_layer(nn.Linear(hidden_dimension, output_dim))

    def forward(self, state, hidden_state, mask):
        x = self.mlp(state)
        x, next_hidden_state = self.rnn(x, hidden_state, mask)
        value = self.value_head(x)
        return value, next_hidden_state