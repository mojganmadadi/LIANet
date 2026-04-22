from curses import window
from shapely import bounds, buffer
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm 
import numpy as np
from datetime import datetime
import math
import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds
from utils import s2_to_rgb, _preprocess_S2
from rasterio.warp import reproject, Resampling
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
            10980, 
            tb=training_bounds_left_top_right_bottom, 
            train_val_key=self.train_val_key,
            patch_size=patch_size,
            exclude_px1_px2=exclude_px1_px2
            )


        # self.samples = []
        # for loc in tqdm(locations, desc=f"Building {train_val_key} samples"):
        #     for sample_idx, time_str in enumerate(s2_tiles):
        #         # print(time_str)
        #         if "BF" not in time_str:
        #             self.samples.append({
        #                 "sample_idx": int(sample_idx),
        #                 "time_str": time_str,
        #                 "location": loc
        #             })
        #         else: continue

        self.min_building_frac = 0.1  # 1%
        self.win_size = 160            # the "big" window before 128 crop
        self.scale = 4                 # label resolution is 4x finer than S2 grid

        # --- replace your current self.samples building loop with this ---
        self.samples = []
        kept_locs = 0
        skipped_locs = 0

        # Open label raster once (much faster than opening for every loc)
        with rasterio.open(os.path.join(self.top_dir, self.labels)) as src_lbl:
            for loc in tqdm(locations, desc=f"Building {train_val_key} samples"):

                # read label window corresponding to the 160x160 S2 window (scaled to label gsd)
                win_lbl = Window(loc[1] * self.scale, loc[0] * self.scale,
                                self.win_size * self.scale, self.win_size * self.scale)

                lbl = src_lbl.read(1, window=win_lbl)

                # Compute building fraction in this window.
                # If lbl is {0,1} (or {0,255}), adapt accordingly:
                if lbl.dtype != np.bool_:
                    # Common cases:
                    # - 0/1 integer mask: mean gives fraction
                    # - 0/255 uint8 mask: (lbl > 0).mean gives fraction
                    building_frac = (lbl > 0).mean()
                else:
                    building_frac = lbl.mean()

                if building_frac <= self.min_building_frac:
                    skipped_locs += 1
                    continue

                kept_locs += 1

                # Only now expand across time / seasons
                for sample_idx, time_str in enumerate(s2_tiles):
                    if "BF" in time_str:
                        continue
                    self.samples.append({
                        "sample_idx": int(sample_idx),
                        "time_str": time_str,
                        "location": loc,
                        "building_frac_160": float(building_frac),  # optional, helpful for debugging
                    })

        np.random.shuffle(self.samples)
        print(f"Kept {kept_locs}/{kept_locs + skipped_locs} locations with >{int(self.min_building_frac*100)}% buildings")
        print(f"{len(self.samples)} {train_val_key} samples after exclusion + building filter")

        np.random.shuffle(self.samples)
        print(f"{len(self.samples)} {train_val_key} samples after exclusion")

    def _get_dt_properties(self, time_str):

        capture_time = os.path.splitext(os.path.basename(time_str))[0]
        dt = datetime.strptime(capture_time, "%Y%m%dT%H%M%S")

        t0 = datetime(2015, 1, 1)
        delta = (dt - t0).total_seconds() / 86400.0  # days since t0

        # day-of-year
        doy = dt.timetuple().tm_yday  # 1..365/366
        doy_norm = (doy - 1) / 365.0
        doy_sin = math.sin(2 * math.pi * doy_norm)
        doy_cos = math.cos(2 * math.pi * doy_norm)

        return {"file_name": time_str,"delta_days": delta, "doy_sin": doy_sin, "doy_cos": doy_cos,}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        dt_properties = self._get_dt_properties(s["time_str"])
        if self.train_val_key == "val":
            y_off, x_off = 16, 16
        else:
            y_off = np.random.randint(0, self.buffer + 1)
            x_off = np.random.randint(0, self.buffer + 1)

        s2_path = os.path.join(self.top_dir, self.s2_tiles[s["sample_idx"]])
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
            label = (label > 0).astype(np.int64)  # (512,512) {0.0, 1.0}

        return {
            "timestamp": torch.tensor(dt_properties["delta_days"], dtype=torch.float32),
            "time_str": s["time_str"],
            "x_s2": x_off + s["location"][1],
            "y_s2": y_off + s["location"][0],
            "s2data": img,
            "label": torch.tensor(label, dtype=torch.int64)
        }

class PASTIS(Dataset):
    def __init__(self,
                 top_dir,
                 s2_tiles,
                 labels,
                 train_val_key,
                 val_folds,
                 ):


        self.top_dir = top_dir # "/home/user/data_shared"
        self.s2_tiles = s2_tiles # "T32ULU"
        self.labels_path = os.path.join(top_dir, labels, self.s2_tiles)
        self.metadata_path = os.path.join(top_dir, labels, "metadata.geojson")
        self.train_val_key = train_val_key
        self.val_folds = val_folds # [2,3] list of integers from 1 to 5

        # take the first image in the tiles path as reference
        self.list_of_s2_tiles = os.listdir(os.path.join(self.top_dir, self.s2_tiles))
        ref_img_path = os.path.join(self.top_dir, self.s2_tiles, self.list_of_s2_tiles[0])
        ref_img = rasterio.open(ref_img_path)
        ref_transform = ref_img.transform
        metadata_gdf = gpd.read_file(self.metadata_path)

        patch_id_col = "ID_PATCH"

        if "fold" in metadata_gdf.columns:
            fold_col = "fold"
        elif "Fold" in metadata_gdf.columns:
            fold_col = "Fold"
        else:
            raise ValueError("No fold column found in metadata.")
        t0 = datetime(2015, 1, 1)
        tile_doy_dates = []
        
        self.samples = []
        for t in tqdm(self.list_of_s2_tiles, desc=f"Building {train_val_key} image label pairs"):
            tile_ds = rasterio.open(os.path.join(self.top_dir, self.s2_tiles, t))
             
            date = t.split(".tif")[0]
            dt = datetime.strptime(date , "%Y%m%dT%H%M%S")   
            doy = (dt - t0).total_seconds() / 86400.0  # days since t0
            for f in os.listdir(self.labels_path):
                if f.endswith(".tif"):
                    patch_id_str = f.split("_")[1].split(".")[0]  # e.g., "TARGET_40000.tif"
                    if metadata_gdf[patch_id_col].dtype.kind in "iu" and patch_id_str.isdigit():
                        patch_id = int(patch_id_str)
                    else:
                        patch_id = patch_id_str
                    # open the labels with rasterio
                    label_ds = rasterio.open(os.path.join(self.labels_path, f))
                    label_patch = label_ds.read().squeeze()
                    row_min, col_min = rasterio.transform.rowcol(ref_transform, label_ds.bounds.left, label_ds.bounds.top)
                    image_patch = tile_ds.read(window=Window(col_min, row_min, 128, 128)) 
                    fold = metadata_gdf[metadata_gdf[patch_id_col] == patch_id][fold_col].values[0]
                    
                    if self.train_val_key == "train" and fold not in self.val_folds:
                            self.samples.append({
                                "doy": doy,
                                "x": col_min,
                                "y": row_min,
                                "patch_id": patch_id,
                                "fold": fold,
                                "s2_img_patch": image_patch,
                                "label": label_patch,
                                })
                    elif self.train_val_key == "val" and fold in self.val_folds:
                            self.samples.append({
                                "doy": doy,
                                "x": col_min,
                                "y": row_min,
                                "patch_id": patch_id,
                                "fold": fold,
                                "s2_img_patch": image_patch,
                                "label": label_patch,  # (128,128) uint8 with values {0,1}
                                })
                    else: continue
        print(f"Found {len(self.samples)} samples for {train_val_key} with >0% burned area")
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        s = self.samples[idx]
        return {
            "delta_days": torch.tensor(s["doy"], dtype=torch.float32),
            "x_s2": s["x"],
            "y_s2": s["y"],
            "s2data": _preprocess_S2(s["s2_img_patch"]),
            "label": s["label"],
            "fold": s["fold"],
            "patch_id": s["patch_id"]
        }
    
class BurnScars(Dataset):
    def __init__(self,
                 top_dir,
                 s2_tiles,
                 labels,
                 train_val_key,
                 ):


        self.top_dir = top_dir # "/home/user/data_shared"
        # self.s2_tiles = s2_tiles # "T11SMT"
        self.labels_path = os.path.join(top_dir, labels, s2_tiles)
        # self.metadata_path = metadata_path
        self.train_val_key = train_val_key
        # self.labels = []

        if self.train_val_key == "train":
            img_label_pairs = {
                'aligned_subsetted_512x512_HLS.S30.T11SMT.2019294.v1.4.mask.tif':'20190504T182921.tif',
                'aligned_subsetted_512x512_HLS.S30.T11SMT.2018249.v1.4.mask.tif':'20180926T183121.tif',
                'aligned_subsetted_512x512_HLS.S30.T11SMT.2018154.v1.4.mask.tif':'20180504T182919.tif',
                'aligned_subsetted_512x512_HLS.S30.T11SMT.2020194.v1.4.mask.tif':'20200423T182909.tif',
                'aligned_subsetted_512x512_HLS.S30.T11SMT.2020289.v1.4.mask.tif':'20200930T183109.tif',
                'aligned_subsetted_512x512_HLS.S30.T11SMT.2020249.v1.4.mask.tif':'20200930T183109.tif',
                'aligned_subsetted_512x512_HLS.S30.T11SMT.2020309.v1.4.mask.tif':'20200930T183109.tif',
                }
        elif self.train_val_key == "val":
            img_label_pairs = {
                'aligned_subsetted_512x512_HLS.S30.T11SMT.2019309.v1.4.mask.tif':'20191006T183229.tif', # Validation
                'aligned_subsetted_512x512_HLS.S30.T11SMT.2021248.v1.4.mask.tif':'20210826T182919.tif', # Validation
            }


        t0 = datetime(2015, 1, 1)
        tile_doy_dates = []
        xy_offset_points = [i for i in range(0, 1536, 128)]
        self.samples = []
        for t in tqdm(img_label_pairs.keys(), desc=f"Building {train_val_key} image label pairs"):
            # read the label with corresponding 
            tile_containing_img = img_label_pairs[t]
            tile_ds = rasterio.open(os.path.join(self.top_dir, "T11SMT" ,tile_containing_img))
             
            date = tile_containing_img.split(".tif")[0]
            dt = datetime.strptime(date , "%Y%m%dT%H%M%S")   
            doy = (dt - t0).total_seconds() / 86400.0  # days since t0
            
            label_ds = rasterio.open(os.path.join(self.labels_path, t))
            # iterate through xy_offset_points
            for x in xy_offset_points:
                for y in xy_offset_points:
                    #read labels with this x and y as the top left corner, and 128x128 window size
                    label_window = Window(x, y, 128, 128)
                    lable_patch = label_ds.read(window=label_window)
                    left, bottom, right, top = window_bounds(label_window, transform=label_ds.transform)
                    s2_window = from_bounds(left, bottom, right, top, transform=tile_ds.transform)
                    # read the same area in image
                    s2_patch = tile_ds.read(window=s2_window)
                    assert lable_patch.shape[1] == 128 and lable_patch.shape[2] == 128, f"S2 patch shape {s2_patch.shape} does not match label patch shape {lable_patch.shape}"
                    assert s2_patch.shape[1] == 128 and s2_patch.shape[2] == 128, f"S2 patch shape {s2_patch.shape} does not match label patch shape {lable_patch.shape}"
                    lable_patch[lable_patch == -1] = 0

                    if lable_patch.sum() == 0: # do not use labels with only non-burned pixels
                        continue
                    else:
                        #fill -1 values in label with 0 (non-burned)
                        burned_pixel_count = (lable_patch > 0).sum()
                        self.samples.append({
                            "doy": doy,
                            "x": int(s2_window.col_off),
                            "y": int(s2_window.row_off),
                            "s2_img_patch": s2_patch,
                            "label": lable_patch.squeeze(),  # (128,128) uint8 with values {0.0, 1.0}
                            "burned_pixel_count": burned_pixel_count,
                            })

                    
            
        np.random.shuffle(self.samples)
        print(f"Found {len(self.samples)} samples for {train_val_key} with >0% burned area")
        print(f"Average burned pixel count across samples: {np.mean([s['burned_pixel_count'] for s in self.samples])}")
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        s = self.samples[idx]
        return {
            "delta_days": torch.tensor(s["doy"], dtype=torch.float32),
            "x_s2": s["x"],
            "y_s2": s["y"],
            "s2data": _preprocess_S2(s["s2_img_patch"]),
            "label": s["label"],
            "burned_pixel_count": s["burned_pixel_count"],
        }
    
