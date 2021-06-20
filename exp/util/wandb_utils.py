from typing import Tuple
from typing import Union

import torch
from wandb.data_types import Node
from wandb.wandb_torch import TorchGraph


def make_wandb_graph(model: torch.nn.Module, input_tensor: Union[torch.Tensor, Tuple[int, ...]], device=None):
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


def _is_numbered(name: str):
    try:
        int(name.split('/')[-1])
        return True
    except:
        return False


def _make_simple_wandb_graph(model, model_input):
    """
    Create a wandb graph of a pytorch model!
    - calls the internal pytorch SummaryWriter functions,
      these APIs are not publicly exposed and may be changed!
      (tested in version v1.9.0 of PyTorch)
    """

    # make graph using pytorch tensorboard hook
    from torch.utils.tensorboard._pytorch_graph import graph
    graph_def, graph_meta = graph(model, model_input)

    # create the wandb graph
    graph = TorchGraph()

    # TODO: a better solution is to construct a DiGraph
    #       and merge nodes together that only have one neighbour

    # extract the nodes
    for n in graph_def.node:
        # graph_def nodes have the following attributes:
        # - name: str (unique)           | eg. `VGG/Sequential[features]/ReLU[18]/input.37`
        # - op: str                      | eg. `prim::GetAttr`, `aten::_convolution`
        # - input: List[str]             | names of other nodes
        # - device: ???                  | ???
        # - experimental_debug_info: ??? | ???
        # - attr: Dict[str, Any]         | sometimes has `_output_shapes: list[shape.dim.size.int]`

        # do not visit primatives
        # do not visit individual weights which end in a number eg. "VGG/Sequential[classifier]/Linear[6]/866"
        if n.name.startswith('prim::') or _is_numbered(n.name):
            continue

        # get output shape
        output_shape = None
        if '_output_shapes' in n.attr:
            output_shape = n.attr['_output_shapes'].list.shape
            assert len(output_shape) == 1
            output_shape = tuple(d.size for d in output_shape[0].dim)

        # create node
        node = Node(
            id=n.name,
            name=n.name,
            class_name=n.op,
            size=None,  # input shape?
            parameters=None,
            output_shape=output_shape,
            is_output=False,
            num_parameters=None,
            node=None,
        )

        # save the node
        assert node.id not in graph.nodes_by_id
        graph.add_node(node)

    # create the links between nodes
    for n in graph_def.node:
        # skip the node if it does not exist
        if n.name not in graph.nodes_by_id:
            continue
        to_name = n.name
        to_node = graph.nodes_by_id[to_name]
        # get inputs to node
        for from_name in n.input:
            # skip the node if it does not exist
            if from_name not in graph.nodes_by_id:
                # print(f'skipping adding edge from: {repr(from_name)} to {repr(to_name)}')
                continue
            # add the edge to the new node
            from_node = graph.nodes_by_id[from_name]
            graph.add_edge(from_node, to_node)

    # done!
    return graph
