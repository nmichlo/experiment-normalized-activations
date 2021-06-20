from typing import Tuple
from typing import Union

import torch
from wandb.wandb_torch import TorchGraph


def make_wandb_graph(model: torch.nn.Module, input_tensor: Union[torch.Tensor, Tuple[int, ...]], device=None):
    """
    Helper function to generate a wandb model graph.
    :param model: the model to generate the graph for.
    :param input_tensor: input tensor, or an input observation shape (excluding the batch dimension).
    :param device: the device to move the model to when feeding forward.
    :return:
    """
    # sample values
    if isinstance(input_tensor, tuple):
        input_tensor = torch.randn(1, *input_tensor, dtype=torch.float32, device=device)
    # update device
    if device is not None:
        model = model.to(device=device)
        input_tensor = input_tensor.to(device=device)
    # hook model
    graph = TorchGraph.hook_torch(model)
    # feed forward & send loss backward
    model(input_tensor.detach()).mean().backward()
    # return the graph
    return graph
