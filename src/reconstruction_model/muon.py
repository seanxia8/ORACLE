import os
import sys

import torch
from torch import Tensor


def _compile_if_supported(fn):
    if os.environ.get("RECONSTRUCTION_DISABLE_TORCH_COMPILE") == "1":
        return fn
    if sys.version_info >= (3, 13):
        return fn
    return torch.compile(fn)


@_compile_if_supported
def orthogonalise_via_newtonschulz5(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm(dim=(-1, -2), keepdim=True) + eps
    if G.size(-2) > G.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A @ X + c * A @ A @ X
        X = a * X + B
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """For more information about the Muon optimiser: https://kellerjordan.github.io/posts/muon/"""

    def __init__(
        self,
        params,
        lr: float = 0.001,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires positive momentum value.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]
                # mt = mt-1 + (1 - mu ) * (grad - mt-1)
                # -> mt = mu * mt-1 + (1 - mu) * grad which is the standard EMA momentum update
                momentum_buffer.lerp_(grad, 1.0 - group["momentum"])
                # g = g + mu * (mt - g)
                # -> g = g * (1 - mu) + mu * mt
                grad = (
                    grad.lerp_(momentum_buffer, group["momentum"])
                    if group["nesterov"]
                    else momentum_buffer
                )
                grad = orthogonalise_via_newtonschulz5(grad, steps=group["ns_steps"])
                p.sub_(
                    grad,
                    alpha=group["lr"] * max(1, (p.size(-2) / p.size(-1)) ** 0.5),
                )
        return loss
