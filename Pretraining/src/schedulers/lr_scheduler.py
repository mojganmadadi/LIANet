import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine annealing.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of warmup epochs.
        max_epochs (int): Total number of epochs.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = float(self.last_epoch + 1) / float(self.warmup_epochs)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [base_lr * cosine_factor for base_lr in self.base_lrs]
        