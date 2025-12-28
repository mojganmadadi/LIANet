import torch
import torch.nn as nn
import torch.nn.functional as F

_C1_T = torch.tensor(-7046029288634856825, dtype=torch.int64)
_C2_T = torch.tensor(-4417276706812531889, dtype=torch.int64)

def _fast_hash_2d(ix, iy, seed, T):
    c1 = _C1_T.to(device=ix.device)
    c2 = _C2_T.to(device=ix.device)
    h = torch.bitwise_xor(ix.to(torch.int64) * c1, iy.to(torch.int64) * c2)
    h = torch.bitwise_xor(h, seed.to(torch.int64))
    return torch.remainder(h, T).to(torch.int64)



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
