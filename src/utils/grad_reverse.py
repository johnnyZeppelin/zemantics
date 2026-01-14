# src/utils/grad_reverse.py
from __future__ import annotations
import torch
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Gradient Reversal Layer (GRL).
    Forward: identity
    Backward: multiply gradient by -alpha
    """
    return _GradReverse.apply(x, alpha)
