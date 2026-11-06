import torch
from torch import nn

def get_torch_device() -> torch.device:
    """Select the best available PyTorch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def init_orthogonal_layer(layer, gain: int|float =1):
    """
       W <- orthogonal matrix * gain
       b <- 0
    """
    nn.init.orthogonal_(layer.weight, gain=gain)

    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

    return layer

def create_mlp_layers(input_dim, hidden_dimension, num_intermediate_layers=1):
    add_intermediate_layer = lambda: nn.Sequential(init_orthogonal_layer(nn.Linear(hidden_dimension, hidden_dimension),
                                                                         gain=nn.init.calculate_gain('relu')),
                                                   nn.ReLU(),
                                                   nn.LayerNorm(hidden_dimension))

    return nn.Sequential(
        init_orthogonal_layer(nn.Linear(input_dim, hidden_dimension), nn.init.calculate_gain('relu')),
        nn.ReLU(),
        nn.LayerNorm(hidden_dimension),
        *(add_intermediate_layer() for _ in range(num_intermediate_layers)),
    )

def init_hidden_state(batch_size, hidden_dimension, recurrent_layers = 1):
    # format as [B (batch size), L (layers), H (hidden dimensions)] due to RNN Layer requirements
    return torch.zeros(batch_size, recurrent_layers, hidden_dimension, device=get_torch_device())

def init_rnn_mask(batch_size, mask_dimension):
    return torch.zeros(batch_size, mask_dimension, device=get_torch_device())

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape