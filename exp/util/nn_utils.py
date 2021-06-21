from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

from pytorch_lightning.utilities import AttributeDict
from torch import nn


def replace_modules(
    module: nn.Module,
    visit_type: Union[Type[nn.Module], Sequence[Type[nn.Module]]],
    visit_fn: Union[Type[nn.Module], Callable[[nn.Module, nn.Module, str], Optional[nn.Module]]],
    is_instance: bool = False,
    verbose: bool = True,
):
    # normalise replace type
    if issubclass(visit_type, nn.Module):
        visit_type = [visit_type]
    visit_type = set(visit_type)

    # if we should match types or instances of those types
    if is_instance:
        visit_type = tuple(visit_type)

    # normalise replace fn
    if type(visit_fn) == type:
        if issubclass(visit_fn, nn.Module):
            _fn, visit_fn = visit_fn, lambda parent, child, key: _fn()

    # count visited or replaced layers
    visit_count = 0
    replace_count = 0

    # recursive replace
    def _replace(parent: nn.Module):
        nonlocal visit_count, replace_count
        for child_name, child_module in parent.named_children():
            # should replace
            if is_instance:
                visit = isinstance(child_module, visit_type)
            else:
                visit = type(child_module) in visit_type
            # do replacement
            if visit:
                new_module = visit_fn(parent, child_module, child_name)
                visit_count += 1
                if new_module is not None:
                    setattr(parent, child_name, new_module)
                    replace_count += 1
            # recurse children
            _replace(child_module)

    # replace everything
    _replace(module)

    # debug
    if verbose:
        l_red, reset = '\033[91m', '\033[0m'
        if is_instance:
            print(f'{l_red}visited #{visit_count} layers that are instances of {sorted(t.__name__ for t in visit_type)} and replaced #{replace_count}{reset}')
        else:
            print(f'{l_red}visited #{visit_count} layers that are of the types {sorted(t.__name__ for t in visit_type)} and replaced #{replace_count}{reset}')

    # return details
    return AttributeDict(
        visit_count=visit_count,
        replace_count=replace_count,
    )


def replace_conv_and_bn(module, conv_class):

    def replace_fn_conv(parent, child, key):
        return conv_class(child.in_channels, child.out_channels, child.kernel_size, child.stride, child.padding, child.dilation, child.groups, hasattr(child, 'bias'))

    def replace_bn(parent, child, key):
        return nn.Identity()

    replace_modules(module, nn.Conv2d, replace_fn_conv, is_instance=True)
    replace_modules(module, nn.BatchNorm2d, replace_bn, is_instance=True)
