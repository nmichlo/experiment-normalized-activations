import torch


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


class SwishFunction(torch.autograd.Function):
    # This uses less memory than the standard implementation,
    # by re-computing the gradient on the backward pass

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        y = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


# ========================================================================= #
# Module & Function                                                         #
# ========================================================================= #


def swish(tensor: torch.Tensor):
    return SwishFunction.apply(tensor)


class Swish(torch.nn.Module):
    def forward(self, tensor: torch.Tensor):
        return SwishFunction.apply(tensor)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
