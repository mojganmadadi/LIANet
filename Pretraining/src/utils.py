import numpy as np

def s2_to_rgb(s2data):
    """Convert Sentinel-2 data to RGB for visualization."""
    # Assuming s2data is of shape (C, H, W) and C >= 3
    if s2data.shape[0] == 12:
        R = s2data[3, :, :]
        G = s2data[2, :, :]
        B = s2data[1, :, :]
    else:
        R = s2data[2, :, :]
        G = s2data[1, :, :]
        B = s2data[0, :, :]

    tensor = np.stack([R, G, B], axis=-1)
    tensor = np.clip(tensor*4,0,1)

    return tensor

def _preprocess_S2(patch):
    patch = patch.astype(np.float32)
    patch = patch - 1000
    patch = patch / 10000
    patch = np.clip(patch, 0, 1)
    return patch
