import torch
import torch.nn as nn
import torch.nn.functional as F

# 64-bit mixing constants
_C1 = 0x9E3779B185EBCA87
_C2 = 0xC2B2AE3D27D4EB4F

def _fast_hash_2d(ix: torch.Tensor, iy: torch.Tensor, seed: torch.Tensor, T: int):
    """
    ix, iy: int64 tensors
    seed:  scalar int64 tensor (per level), same device as ix/iy
    return: int64 indices in [0, T)
    """
    h = ix.mul(_C1) ^ iy.mul(_C2) ^ seed
    return torch.remainder(h, T).long()

def group_norm(c: int) -> nn.GroupNorm:
    """
    Returns a GroupNorm layer with a reasonable number of groups 
    based on the number of channels 'c'.
    Tuned for feature maps in range 64–2048 channels.
    """
    if c <= 64:
        groups = 4
    elif c <= 128:
        groups = 8
    elif c <= 256:
        groups = 16
    elif c <= 512:
        groups = 32
    elif c <= 1024:
        groups = 32  # 64 works too but 32 is safe for memory
    else:  # up to 2048 or more
        groups = 32  # you can increase to 64 if channels are huge

    return nn.GroupNorm(num_groups=groups, num_channels=c)
