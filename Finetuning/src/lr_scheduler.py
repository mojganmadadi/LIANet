import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0.0, last_epoch=-1):
        """
        Args:
            optimizer: Optimizer (e.g., torch.optim.Adam)
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of epochs
            min_lr: Minimum learning rate after decay
            last_epoch: The index of the last epoch (for resuming training)
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # linearly scale up during warmup
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]

        # cosine decay after warmup
        progress = (self.last_epoch - self.warmup_epochs) / (
            self.max_epochs - self.warmup_epochs
        )
        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]