from torch import nn

class RNNLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        recurrent_layers=1,
    ):
        """
        original MAPPO implementation uses GRU, so we're trying to stay close to it
        the default number of recurrent layers is 1, so we'll keep it that way to
        maintain some simplicity.

        :param input_dim:
        :param hidden_dim:
        :param recurrent_layers: default 1
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.recurrent_layers = recurrent_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=recurrent_layers,
        )

        # Match MAPPO initialization
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)

            elif "weight" in name:
                nn.init.orthogonal_(param)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x,
        hidden_state,
        mask,
    ):
        # hidden_state: [batch, recurrent_layers, hidden_dim]
        # mask: [batch, 1]
        hidden_state = hidden_state * mask.unsqueeze(-1)

        # GRU expects: [recurrent_layers, batch, hidden_dim]
        hidden_state = hidden_state.transpose(0, 1).contiguous()

        # x: [batch, hidden_dim] but GRU expects: [sequence, batch, hidden_dim]
        x = x.unsqueeze(0)

        output, hidden_state = self.gru(
            x,
            hidden_state,
        )

        # Remove sequence dimension
        output = output.squeeze(0)

        # Back to:
        # [batch, recurrent_layers, hidden_dim] -> original form
        hidden_state = hidden_state.transpose(0, 1).contiguous()

        output = self.norm(output)

        return output, hidden_state