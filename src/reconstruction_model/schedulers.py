import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def cosine_scheduler_with_linear_warmup(
    optimiser,
    num_warmup_steps: int,
    total_steps: int,
    last_epoch: int = -1,
):
    main_cycle_steps = float(max(1, total_steps - num_warmup_steps))

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(0.0, float(current_step) / float(max(1, num_warmup_steps)))

        progress = float(current_step - num_warmup_steps) / main_cycle_steps
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimiser, lr_lambda, last_epoch)
