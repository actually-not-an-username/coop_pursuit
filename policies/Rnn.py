import torch
from torch import nn
from torch.jit.frontend import NotSupportedError


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
        if x.dim() == 2:
            return self.forward_2d(x, hidden_state, mask)
        elif x.dim() == 3:
            return self.forward_3d(x, hidden_state, mask)
        else:
            raise NotSupportedError



    def forward_2d(self, x, hidden_state, mask):
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

    def forward_3d(self, x, hidden_state, mask):
        # when doing recurrent training the shapes of the tensors are
        # different from the original single episode training
        chunk_length, batch_size, _ = x.shape
        outputs = []

        # hidden_state: [batch, recurrent_layers, hidden_dim]
        # mask: [recurrent_chunk_length, batch, 1]
        hidden_state = hidden_state.transpose(0, 1).contiguous()

        for chunk in range(chunk_length):
            # ---------------------------------------------
            # Reset hidden state wherever mask[t] == 0
            #
            # hidden_state: [recurrent_layers, batch, hidden_dim]
            # mask: [batch, 1]
            # ---------------------------------------------
            # reshape mask to [1, batch, 1]
            hidden_state = hidden_state * mask[chunk].unsqueeze(0)

            # x_t here is [batch, hidden_dim] but this GRU cell expects [1, batch, hidden_dim]
            x_t = x[chunk].unsqueeze(0)

            output_t, hidden_state = self.gru(
                x_t,
                hidden_state,
            )

            # [1, batch, hidden_dim] -> [batch, hidden_dim]
            output_t = output_t.squeeze(0)
            outputs.append(output_t)

        # Format the output from a list of T (recurrent_chunk_length) tensors to
        # a tensor of shape [chunk_length, batch, hidden_dim]
        output = torch.stack(outputs, dim=0)
        output = self.norm(output)

        # Back to:
        # [batch, recurrent_layers, hidden_dim] -> original form
        hidden_state = (
            hidden_state
            .transpose(0, 1)
            .contiguous()
        )

        return output, hidden_state