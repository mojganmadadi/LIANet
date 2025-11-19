import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm 
import numpy as np

import rasterio
from rasterio.windows import Window


from utils import read_and_normalize_s2, get_sample_locations


class DynamicWorld(Dataset):
    def __init__(self,
                 top_dir,
                 s2_tiles,
                 labels,
                 training_bounds_left_top_right_bottom, # Defining the train region [x_min, y_min, x_max, y_max]: [4000, 0, 5000, 5000]
                 train_val_key,
                 complete_tile_size,
                 patch_size=128,
                 buffer=32
                 ):

        """
        HARDCODED!!!!! dataloader for 128 patch size and a 32 pixel buffer to generate random crops around each patch.
        """

        self.top_dir = top_dir
        self.s2_tiles = s2_tiles
        self.labels = labels
        self.train_val_key = train_val_key
        self.patch_size = patch_size
        self.buffer = buffer

        locations = get_sample_locations(
            complete_tile_size, 
            tb=training_bounds_left_top_right_bottom, 
            train_val_key=self.train_val_key,
            patch_size=patch_size,
            exclude_px1_px2=None
            )

        # handle nodata exclusion here
        nodata_count = 0
        self.samples = []
        for loc in tqdm(locations):
            for _, label_file in enumerate(labels):
                season_index = label_file.split("_")[2]  # assuming format like "dw_20220312_0_03.tif"
                with rasterio.open(os.path.join(top_dir, label_file)) as src:
                    win = Window(loc[1], loc[0], 160, 160)
                    patch = src.read(window=win)
                    counts = np.bincount(patch.flatten(), minlength=8)
                    nodata = np.any(patch==50)

                    if not nodata:
                        self.samples.append({
                            "season_index": int(season_index),
                            "location": loc,
                            "landcover_counts": counts,
                            })
                    else:   
                        nodata_count += 1

        np.random.shuffle(self.samples)

        print(f"Found {len(self.samples)} samples for {train_val_key}")
        print(f"Found {nodata_count} no-data patches for {train_val_key}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        s = self.samples[idx]

        # get random crop location within the 160x160 patch
        if self.train_val_key == "val":
            y_off = 16  # center crop
            x_off = 16  # center crop
        else:
            y_off = np.random.randint(0, self.buffer + 1)  # 0 to 32
            x_off = np.random.randint(0, self.buffer + 1)  # 0 to 32

        s2_path = os.path.join(self.top_dir, self.s2_tiles[s["season_index"]])
        # ---- Read Sentinel-2 raster window & crop ----
        img = read_and_normalize_s2(
            s2_path,
            s,
            x_off,
            y_off,
            self.patch_size,
            win_size=160
        )

        # load label
        with rasterio.open(os.path.join(self.top_dir, self.labels[s["season_index"]])) as src:
            win = Window(s["location"][1], s["location"][0], 160, 160)
            label = src.read(1, window=win)  # HxW, uint8
            label = label[y_off:y_off+128, x_off:x_off+128]  # random crop to 128x128

            # --- 1) Merge rare classes via LUT ---
            # Start with identity LUT for 0..8
            lut = np.arange(9, dtype=np.uint8)

            # Merge rules (edit as you like):
            # Flooded vegetation(3) -> Water(0)
            lut[3] = 0
            # Shrub & scrub(5) -> Grass(2)
            lut[5] = 2
            # Snow & ice(8) -> Bare ground(7)
            lut[8] = 7

            # Apply in one shot
            label = lut[label]

            # Re-index the class labels
            # Old -> New: 0->0, 1->1, 2->2, 4->3, 6->4, 7->5
            reindex_lut = np.full(9, 255, dtype=np.uint8)  # 255 as ignore_index for anything unexpected
            reindex_lut[0] = 0   # water
            reindex_lut[1] = 1   # trees
            reindex_lut[2] = 2   # grass (incl. merged shrub & scrub)
            reindex_lut[4] = 3   # crops
            reindex_lut[6] = 4   # built area
            reindex_lut[7] = 5   # bare ground (incl. merged snow & ice)
            label = reindex_lut[label]

        label = torch.from_numpy(label).long()
            
        return {
            "timestamp": s["season_index"],
            "x_s2": x_off + s["location"][1],
            "y_s2": y_off + s["location"][0],
            "s2data": img,
            "label": label
        }

class MetaCanopyHeights(Dataset):
    def __init__(self,
                 top_dir,
                 s2_tiles,
                 labels,
                 training_bounds_left_top_right_bottom,
                 train_val_key,
                 complete_tile_size,
                 patch_size=128,
                 buffer=32
                 ):
        """
        Dataset for canopy height estimation using Sentinel-2 imagery.

        Args:
            top_dir: Base directory containing the data files.
            s2_tiles: List of Sentinel-2 seasonal image file paths (relative to top_dir).
                      Example: ["S2_0_03.tif", "S2_1_05.tif", "S2_2_09.tif", "S2_3_10.tif"]
            label_file: Path (relative to top_dir) to a single canopy height raster.
            train_bounds_left_top_right_bottom: [x_min, y_min, x_max, y_max] defining training region.
            train_val_key: "train" or "val".
            complete_tile_size: Total pixel width/height of the full tile.
            normalize_labels: If True, normalize canopy heights from [0,43] → [0,1].
        """

        self.top_dir = top_dir
        self.s2_tiles = s2_tiles
        self.labels = labels
        self.train_val_key = train_val_key
        self.patch_size = patch_size
        self.buffer = buffer
        self.max_height_m = 30

        locations = get_sample_locations(
            complete_tile_size, 
            tb=training_bounds_left_top_right_bottom, 
            train_val_key=self.train_val_key,
            patch_size=patch_size,
            exclude_px1_px2=None
            )

        # One label file, multiple seasonal Sentinel-2 tiles
        self.samples = []
        for loc in tqdm(locations, desc=f"Building {train_val_key} samples"):
            for season_index, _ in enumerate(s2_tiles):
                self.samples.append({
                    "season_index": int(season_index),
                    "location": loc
                })

        np.random.shuffle(self.samples)
        print(f"Found {len(self.samples)} samples for {train_val_key}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Random or center crop offs
        if self.train_val_key == "val":
            y_off, x_off = 16, 16
        else:
            y_off = np.random.randint(0, self.buffer + 1)
            x_off = np.random.randint(0, self.buffer + 1)

        s2_path = os.path.join(self.top_dir, self.s2_tiles[s["season_index"]])
        # ---- Read Sentinel-2 raster window & crop ----
        img = read_and_normalize_s2(
            s2_path,
            s,
            x_off,
            y_off,
            self.patch_size,
            win_size=160
        )

        # --- Canopy height label (static) ---
        with rasterio.open(os.path.join(self.top_dir, self.labels)) as src:
            win = Window(s["location"][1], s["location"][0], 160, 160)
            label = src.read(1, window=win)
            label = label[y_off:y_off+128, x_off:x_off+128]
            label = label / self.max_height_m 
        label = torch.from_numpy(label).float()
        label = label.unsqueeze(0)

        return {
            "timestamp": s["season_index"],
            "x_s2": x_off + s["location"][1],
            "y_s2": y_off + s["location"][0],
            "s2data": img,
            "label": label
        }

class BuildingCoverageRaster(Dataset):
    """
    Reads per-pixel building fractions directly from a single-band, aligned
    mbf_fractional.tif. Values expected in [0,1].
    """

    def __init__(self,
                 top_dir,
                 s2_tiles,
                 labels,
                 training_bounds_left_top_right_bottom,
                 train_val_key,
                 complete_tile_size,
                 exclude_px1_px2,
                 patch_size=128,
                 buffer=32):
    

        self.top_dir = top_dir
        self.s2_tiles = s2_tiles
        self.labels = labels
        self.train_val_key = train_val_key
        self.patch_size = patch_size
        self.buffer = buffer


        locations = get_sample_locations(
            complete_tile_size, 
            tb=training_bounds_left_top_right_bottom, 
            train_val_key=self.train_val_key,
            patch_size=patch_size,
            exclude_px1_px2=exclude_px1_px2
            )


        self.samples = []
        for loc in tqdm(locations, desc=f"Building {train_val_key} samples"):
            for season_index, _ in enumerate(s2_tiles):
                self.samples.append({
                    "season_index": int(season_index),
                    "location": loc
                })

        np.random.shuffle(self.samples)
        print(f"{len(self.samples)} {train_val_key} samples after exclusion")

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.train_val_key == "val":
            y_off, x_off = 16, 16
        else:
            y_off = np.random.randint(0, self.buffer + 1)
            x_off = np.random.randint(0, self.buffer + 1)

        s2_path = os.path.join(self.top_dir, self.s2_tiles[s["season_index"]])

        # ---- Read Sentinel-2 raster window & crop ----
        img = read_and_normalize_s2(
            s2_path,
            s,
            x_off,
            y_off,
            self.patch_size,
            win_size=160
        )

        # ---- Read fractional raster window & crop (already aligned) ----
        with rasterio.open(os.path.join(self.top_dir, self.labels)) as src:
            win = Window(s["location"][1], s["location"][0], 160, 160)
            label = src.read(1, window=win)  # (160,160)
            label = label[y_off:y_off + self.patch_size, x_off:x_off + self.patch_size]

        label = torch.from_numpy(label).float()  # (128,128) in [0,1]
        label = label.unsqueeze(0) # add a channel dimension for regression

        return {
            "timestamp": s["season_index"],
            "x_s2": x_off + s["location"][1],
            "y_s2": y_off + s["location"][0],
            "s2data": img,      # (C,128,128)
            "label": label      # (128,128) per-pixel building fraction
        }

class DominantLeafTypeSegmentation(Dataset):
    """
    Segmentation dataset for Dominant Leaf Type (3 classes: 0=no forest, 1=broadleaf, 2=conifers)

    - Follows the same grid sampling logic as other datasets (DynamicWorld, MetaCanopyHeights)
    - No excluded region
    - Validation region defined by validation_bounds_left_top_right_bottom
    - Each sample: Sentinel-2 patch (C,128,128) and label map (128,128)
    """

    def __init__(self,
                 top_dir,
                 s2_tiles,
                 labels,
                 training_bounds_left_top_right_bottom,
                 train_val_key="train",
                 complete_tile_size=5000,
                 patch_size=128,
                 buffer=32
                 ):

        self.top_dir = top_dir
        self.s2_tiles = s2_tiles
        self.labels = labels
        self.train_val_key = train_val_key
        self.patch_size = patch_size
        self.buffer = buffer


        locations = get_sample_locations(
            complete_tile_size, 
            tb=training_bounds_left_top_right_bottom, 
            train_val_key=self.train_val_key,
            patch_size=patch_size,
            exclude_px1_px2=None
            )

        # --- 2) Combine locations with all seasonal image indices ---
        self.samples = []
        for loc in tqdm(locations, desc=f"Building {train_val_key} samples"):
            for season_index, _ in enumerate(s2_tiles):
                self.samples.append({
                    "season_index": int(season_index),
                    "location": loc
                })
        np.random.shuffle(self.samples)
        print(f"Found {len(self.samples)} samples for {train_val_key}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # random/center crop inside 160x160
        if self.train_val_key == "val":
            y_off, x_off = 16, 16  # center crop
        else:
            y_off = np.random.randint(0, self.buffer + 1)
            x_off = np.random.randint(0, self.buffer + 1)

        s2_path = os.path.join(self.top_dir, self.s2_tiles[s["season_index"]])
        # ---- Read Sentinel-2 raster window & crop ----
        img = read_and_normalize_s2(
            s2_path,
            s,
            x_off,
            y_off,
            self.patch_size,
            win_size=160
        )

        # --- Dominant Leaf Type label ---
        with rasterio.open(os.path.join(self.top_dir, self.labels)) as src:
            win = Window(s["location"][1], s["location"][0], 160, 160)
            label = src.read(1, window=win)
            label = label[y_off:y_off + self.patch_size, x_off:x_off + self.patch_size]

        label = torch.from_numpy(label).long()

        return {
            "timestamp": s["season_index"],
            "x_s2": x_off + s["location"][1],
            "y_s2": y_off + s["location"][0],
            "s2data": img,    # (C,128,128)
            "label": label    # (128,128)
        }
    
class BuildingBinaryRaster(Dataset):
    """
    Dataset for co-registered binary building segmentation from mbf_binry.tif.
    - Assumes identical CRS/transform/shape to the Sentinel-2 tiles.
    - Uses the same px1/px2 exclusion and (160->128) window/crop as before.
    - Returns:
        s2data: (C,128,128) float in [0,1]
        label:  (128,128) float with values {0.0, 1.0}
    """
    def __init__(self,
                 top_dir,
                 s2_tiles,
                 labels,
                 training_bounds_left_top_right_bottom,
                 train_val_key,
                 complete_tile_size,
                 exclude_px1_px2,
                 patch_size=128,
                 buffer=32):

        self.top_dir = top_dir
        self.s2_tiles = s2_tiles
        self.labels = labels
        self.train_val_key = train_val_key
        self.patch_size = patch_size
        self.buffer = buffer

        locations = get_sample_locations(
            complete_tile_size, 
            tb=training_bounds_left_top_right_bottom, 
            train_val_key=self.train_val_key,
            patch_size=patch_size,
            exclude_px1_px2=exclude_px1_px2
            )


        self.samples = []
        for loc in tqdm(locations, desc=f"Building {train_val_key} samples"):
            for season_index, _ in enumerate(s2_tiles):
                self.samples.append({
                    "season_index": int(season_index),
                    "location": loc
                })

        np.random.shuffle(self.samples)
        print(f"{len(self.samples)} {train_val_key} samples after exclusion")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.train_val_key == "val":
            y_off, x_off = 16, 16
        else:
            y_off = np.random.randint(0, self.buffer + 1)
            x_off = np.random.randint(0, self.buffer + 1)

        s2_path = os.path.join(self.top_dir, self.s2_tiles[s["season_index"]])
        # ---- Read Sentinel-2 raster window & crop ----
        img = read_and_normalize_s2(
            s2_path,
            s,
            x_off,
            y_off,
            self.patch_size,
            win_size=160
        )

        # The labels are in 2.5 meter gsd, so need to scale the window by 4x
        # --- read binary mask window & crop (already aligned) ---
        with rasterio.open(os.path.join(self.top_dir, self.labels)) as src:
            win = Window(s["location"][1]*4, s["location"][0]*4, 160*4, 160*4)
            label = src.read(1, window=win)
            label = label[y_off*4:y_off*4 + self.patch_size*4, x_off*4:x_off*4 + self.patch_size*4]
        label = torch.from_numpy(label).long()  # (128,128) {0.0, 1.0}

        return {
            "timestamp": s["season_index"],
            "x_s2": x_off + s["location"][1],
            "y_s2": y_off + s["location"][0],
            "s2data": img,
            "label": label
        }
