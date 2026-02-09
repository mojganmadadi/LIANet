import numpy as np
import rasterio
import torch
from rasterio.windows import Window

def _intersects_exclusion(y, x, excl, win_size):
    ymin, xmin, ymax, xmax = excl
    wymin, wymax = y, y + win_size
    wxmin, wxmax = x, x + win_size
    return not ((wxmax <= xmin) or (wxmin >= xmax) or (wymax <= ymin) or (wymin >= ymax))

import torch
import torch.nn.functional as F

def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (0, 2, 3)  # N,H,W over batch
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


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

def read_and_normalize_s2(s2_path, s, x_off, y_off, patch_size, win_size=160):
    """Read and normalize Sentinel-2 data from a given path."""
    import rasterio

    # ---- Read Sentinel-2 window & crop ----
    with rasterio.open(s2_path) as src:
        win = Window(s["location"][1], s["location"][0], win_size, win_size)  # 160x160
        img = src.read(window=win)  # (C,160,160)
        img = np.clip((img - 1000) / 10000.0, 0, 1)
        img = img[:, y_off:y_off + patch_size, x_off:x_off + patch_size]  # (C,128,128)
        img = torch.from_numpy(img).float()

    return img


def get_sample_locations(complete_tile_size, tb, train_val_key, patch_size, exclude_px1_px2):

    locations = []
    griding_windows = patch_size + 32  # patch size + buffer

    if not exclude_px1_px2: 
        for y in range(0, complete_tile_size - griding_windows, griding_windows):
            for x in range(0, complete_tile_size - griding_windows, griding_windows):
                if tb[0] <= x < tb[2] and tb[1] <= y < tb[3]:  # 4000<= col <5000 and 0<= row <5000
                    if train_val_key == "train":
                        locations.append((y, x))
                else: 
                    if train_val_key == "val":
                        locations.append((y, x))
    else:
        (py1, px1), (py2, px2) = exclude_px1_px2
        excl = (min(py1, py2), min(px1, px2), max(py1, py2), max(px1, px2))  # (ymin, xmin, ymax, xmax)
        for y in range(0, complete_tile_size - griding_windows, griding_windows):
            for x in range(0, complete_tile_size - griding_windows, griding_windows):
                is_within_ = (tb[0] <= x < tb[2]) and (tb[1] <= y < tb[3])
                if train_val_key == "train" and not is_within_:
                    continue
                if train_val_key == "val" and is_within_:
                    continue
                if _intersects_exclusion(y, x, excl, griding_windows):
                    continue
                locations.append((y, x))

    return locations