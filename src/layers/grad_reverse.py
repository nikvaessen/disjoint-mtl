########################################################################################
#
# Implement a gradient reversal layer.
#
# Taken from https://github.com/janfreyberg/pytorch-revgrad
#
# Author(s): Nik Vaessen
########################################################################################

import torch as t


########################################################################################
# Function


class InverseGradientFunction(t.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


########################################################################################
# as nn module and functional API

# functional
inverse_gradient = InverseGradientFunction.apply


class InverseGradient(t.nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()

        self._alpha = t.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return inverse_gradient(input_, self._alpha)
