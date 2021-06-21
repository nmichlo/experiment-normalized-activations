import contextlib
import inspect
import warnings
from argparse import Namespace
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import torch
from pytorch_lightning.utilities import AttributeDict
from torch import nn


# def module_apply(
#     module: nn.Module,
#     fn: Callable[[Optional[nn.Module], Optional[str], nn.Module], None],
# ):
#     def _recurse(parent, key, child: nn.Module):
#         fn(parent, key, child)
#         for k, v in child.named_children():
#             _recurse(child, k, v)
#     _recurse(None, None, module)
#     return module


VisitTypesHint = Union[Type[nn.Module], Tuple[Type[nn.Module], ...]]


def iter_submodules(
    module: nn.Module,
    visit_type: Optional[VisitTypesHint] = None,
    visit_instance_of: bool = False,
    visit_recursive: bool = True,
    visit_root: bool = False,
):
    # normalise replace type
    if visit_type is not None:
        if isinstance(visit_type, type):
            if issubclass(visit_type, nn.Module):
                visit_type = [visit_type]
        assert isinstance(visit_type, (list, tuple))
        assert len(visit_type) > 0
        assert all(issubclass(v, nn.Module) for v in visit_type)
        visit_type = tuple(visit_type) if visit_instance_of else set(visit_type)

    # recurse function
    def _recurse(current, key, parent):
        # should visit
        if visit_type is None:
            visit = True
        else:
            if visit_instance_of:
                visit = isinstance(current, visit_type)
            else:
                visit = type(current) in visit_type
        # recurse children
        if (visit_recursive) or (parent is None):
            for child_name, child_module in current.named_children():
                yield from _recurse(current=child_module, key=child_name, parent=current)
        # do visit
        if visit:
            if (visit_root) or (parent is not None):
                yield current, key, parent
    # begin
    yield from _recurse(module, None, None)


def replace_submodules(
    module: nn.Module,
    visit_type: VisitTypesHint,
    modify_fn: Union[Callable[[], Optional[nn.Module]], Callable[[nn.Module, str, nn.Module], Optional[nn.Module]]],
    visit_instance_of: bool = False,
):
    assert visit_type is not None
    # type of function
    num_params = len(inspect.signature(modify_fn).parameters)
    assert num_params in (0, 3)
    if num_params == 0:
        _old_modify_fn, modify_fn = modify_fn, lambda c, n, p: _old_modify_fn()
    # count visited or replaced layers
    visit_count = 0
    replace_count = 0
    # recursive replace
    for child, key, parent in iter_submodules(
        module=module,
        visit_type=visit_type,
        visit_instance_of=visit_instance_of,
        visit_recursive=True,
        visit_root=False,
    ):
        # get the modified module
        new_child = modify_fn(child, key, parent)
        visit_count += 1
        # replace the module
        if new_child is not None:
            setattr(parent, key, new_child)
            replace_count += 1
    # return details
    return Namespace(
        visit_count=visit_count,
        replace_count=replace_count,
    )


def find_submodules(
    module: nn.Module,
    visit_type: VisitTypesHint,
    visit_instance_of: bool = False,
) -> List[nn.Module]:
    layers = []
    for child, key, parent in iter_submodules(
        module=module,
        visit_type=visit_type,
        visit_instance_of=visit_instance_of,
        visit_recursive=True,
        visit_root=False,
    ):
        layers.append(child)
    return layers


@contextlib.contextmanager
def in_out_capture_context(layers: Sequence[nn.Module], mode='in_out') -> List[torch.Tensor]:
    inp_stack = []
    out_stack = []

    # handle mode
    add_inp, add_out, yield_vals = {
        'in_out': (True, True, (inp_stack, out_stack)),
        'out':    (False, True, out_stack),
        'in':     (True, False, inp_stack),
    }[mode]

    # initialise capture hook
    def _hook(module, input, output) -> None:
        if len(inp_stack) >= len(layers): raise RuntimeError('input stack was not cleared before next feed forward')
        if len(out_stack) >= len(layers): raise RuntimeError('output stack was not cleared before next feed forward')
        if add_inp: inp_stack.append(input)
        if add_out: out_stack.append(output)

    # register hooks
    handles = [layer.register_forward_hook(_hook) for layer in layers]

    # yield stack
    try:
        yield yield_vals
    finally:
        if len(inp_stack) != 0: warnings.warn('input stack was not cleared before context was exited.')
        if len(out_stack) != 0: warnings.warn('output stack was not cleared before context was exited.')
        inp_stack.clear()
        out_stack.clear()
        # unregister hooks
        for handle in handles:
            handle.remove()


def replace_conv_and_bn(module, conv_class):
    def replace_fn_conv(child, key, parent):
        return conv_class(child.in_channels, child.out_channels, child.kernel_size, child.stride, child.padding, child.dilation, child.groups, hasattr(child, 'bias'))
    def replace_bn(child, key, parent):
        return nn.Identity()
    replaced_conv = replace_submodules(module, nn.Conv2d, replace_fn_conv, visit_instance_of=True)
    replaced_bn = replace_submodules(module, nn.BatchNorm2d, replace_bn, visit_instance_of=True)
    return replaced_conv, replaced_bn
