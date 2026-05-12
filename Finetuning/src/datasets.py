import json
from tkinter.messagebox import IGNORE
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


class BuildingBinaryRaster(Dataset):
    def __init__(
        self,
        top_dir,
        s2_tiles,
        labels,
        train_val_key,
        samples_dir=None,
        patch_size=128,
        label_scale=4,
    ):
        self.top_dir = top_dir
        self.s2_tile_name = s2_tiles
        self.label_name = labels
        self.train_val_key = train_val_key
        self.patch_size = patch_size
        self.label_scale = label_scale

        if samples_dir is None:
            samples_dir = top_dir

        samples_path = os.path.join(
            samples_dir,
            f"{self.s2_tile_name}_{train_val_key}_samples_10perc.json"
        )

        with open(samples_path, "r") as f:
            self.samples = json.load(f)

        print(
            f"{len(self.samples)} {train_val_key} samples loaded "
            f"from {samples_path}"
        )

    def _get_dt_properties(self, time_str):
        capture_time = os.path.splitext(os.path.basename(time_str))[0]
        dt = datetime.strptime(capture_time, "%Y%m%dT%H%M%S")

        t0 = datetime(2015, 1, 1)
        delta_days = (dt - t0).total_seconds() / 86400.0

        return delta_days

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        delta_days = self._get_dt_properties(s["time_str"])

        s2_path = os.path.join(
            self.top_dir,
            self.s2_tile_name,
            s["time_str"]
        )

        label_path = os.path.join(
            self.top_dir,
            "masks",
            "BuildingFootprints",
            self.label_name
        )

        with rasterio.open(s2_path) as src:
            win = Window(
                s["x"],
                s["y"],
                self.patch_size,
                self.patch_size,
            )
            img = src.read(window=win).astype(np.float32)

        scale = self.label_scale

        with rasterio.open(label_path) as src:
            win = Window(
                s["x"] * scale,
                s["y"] * scale,
                self.patch_size * scale,
                self.patch_size * scale,
            )
            label = src.read(1, window=win)
            label = (label > 0).astype(np.int64)

        return {
            "delta_days": torch.tensor(delta_days, dtype=torch.float32),
            "time_str": s["time_str"],
            "x_s2": s["x"],
            "y_s2": s["y"],
            "s2data": _preprocess_S2(img),
            "label": label,
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
    
class BuildingCoverageRaster(Dataset):
    def __init__(
        self,
        top_dir,
        s2_tiles,
        labels,
        train_val_key,
        samples_dir=None,
        patch_size=128,
        label_scale=4,
    ):
        self.top_dir = top_dir
        self.s2_tile_name = s2_tiles
        self.label_name = labels
        self.train_val_key = train_val_key
        self.patch_size = patch_size
        self.label_scale = label_scale

        if samples_dir is None:
            samples_dir = top_dir

        samples_path = os.path.join(
            samples_dir,
            f"{self.s2_tile_name}_{train_val_key}_samples_10perc.json"
        )

        with open(samples_path, "r") as f:
            self.samples = json.load(f)

        print(
            f"{len(self.samples)} {train_val_key} samples loaded "
            f"from {samples_path}"
        )

    def _get_dt_properties(self, time_str):
        capture_time = os.path.splitext(os.path.basename(time_str))[0]
        dt = datetime.strptime(capture_time, "%Y%m%dT%H%M%S")

        t0 = datetime(2015, 1, 1)
        delta_days = (dt - t0).total_seconds() / 86400.0

        return delta_days

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        delta_days = self._get_dt_properties(s["time_str"])

        s2_path = os.path.join(
            self.top_dir,
            self.s2_tile_name,
            s["time_str"]
        )

        label_path = os.path.join(
            self.top_dir,
            "masks",
            "BuildingFootprints",
            self.label_name
        )

        with rasterio.open(s2_path) as src:
            win = Window(
                s["x"],
                s["y"],
                self.patch_size,
                self.patch_size,
            )
            img = src.read(window=win).astype(np.float32)

        scale = self.label_scale

        with rasterio.open(label_path) as src:
            scale = 4

            win = Window(
                s["x"] * scale,
                s["y"] * scale,
                self.patch_size * scale,
                self.patch_size * scale,
            )

            label_2p5m = src.read(1, window=win)
            label_2p5m = (label_2p5m > 0).astype(np.float32)  # (512, 512)

            # Convert 2.5 m binary mask to 10 m building density
            # (512, 512) -> (128, 4, 128, 4) -> (128, 128)
            building_density = label_2p5m.reshape(
                self.patch_size, scale,
                self.patch_size, scale
            ).mean(axis=(1, 3)).astype(np.float32)
            # print(building_density)
        return {
            "delta_days": torch.tensor(delta_days, dtype=torch.float32),
            "time_str": s["time_str"],
            "x_s2": s["x"],
            "y_s2": s["y"],
            "s2data": _preprocess_S2(img),
            "label": building_density,
        }

class PASTIS(Dataset):
    def __init__(self,
                 top_dir,
                 s2_tiles,
                 labels,
                 train_val_key,
                 val_folds,
                 num_classes=19,      # classes 0–18
                 ignore_index=255,
                 compute_weights=True):


        
        self.top_dir = top_dir # "/home/user/data_shared"
        self.s2_tiles = s2_tiles # "T32ULU"
        self.labels_path = os.path.join(top_dir, labels, self.s2_tiles)
        self.metadata_path = os.path.join(top_dir, labels, "metadata.geojson")
        self.train_val_key = train_val_key
        self.val_folds = val_folds # [2,3] list of integers from 1 to 5
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        class_counts = np.zeros(num_classes, dtype=np.int64)

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
        
        self.samples = []
        for t in tqdm(self.list_of_s2_tiles, desc=f"Building {train_val_key} image label pairs"):
            tile_ds = rasterio.open(os.path.join(self.top_dir, self.s2_tiles, t))

            date = t.split(".tif")[0]
            dt = datetime.strptime(date, "%Y%m%dT%H%M%S")
            doy = (dt - t0).total_seconds() / 86400.0

            for f in os.listdir(self.labels_path):
                if f.endswith(".tif"):
                    patch_id_str = f.split("_")[1].split(".")[0]

                    if metadata_gdf[patch_id_col].dtype.kind in "iu" and patch_id_str.isdigit():
                        patch_id = int(patch_id_str)
                    else:
                        patch_id = patch_id_str

                    label_ds = rasterio.open(os.path.join(self.labels_path, f))
                    label_patch = label_ds.read().squeeze()

                    assert not np.isnan(label_patch).any(), "NaNs found in labels"

                    valid = ((label_patch >= 0) & (label_patch <= 19)) | (label_patch == 255)
                    assert valid.all(), f"Unexpected labels: {np.unique(label_patch[~valid])}"

                    # Map void class 19 to ignore
                    label_patch[label_patch == 19] = ignore_index

                    row_min, col_min = rasterio.transform.rowcol(
                        ref_transform,
                        label_ds.bounds.left,
                        label_ds.bounds.top
                    )

                    image_patch = tile_ds.read(window=Window(col_min, row_min, 128, 128))

                    fold = metadata_gdf[
                        metadata_gdf[patch_id_col] == patch_id
                    ][fold_col].values[0]

                    use_sample = (
                        (self.train_val_key == "train" and fold not in self.val_folds)
                        or
                        (self.train_val_key == "val" and fold in self.val_folds)
                    )

                    if use_sample:
                        self.samples.append({
                            "doy": doy,
                            "x": col_min,
                            "y": row_min,
                            "patch_id": patch_id,
                            "fold": fold,
                            "s2_img_patch": image_patch,
                            "label": label_patch,
                        })

                        # Count class pixels only for training set
                        if compute_weights and self.train_val_key == "train":
                            mask = label_patch != ignore_index
                            vals, cnts = np.unique(label_patch[mask], return_counts=True)

                            for v, c in zip(vals, cnts):
                                if 0 <= v < num_classes:
                                    class_counts[int(v)] += int(c)

        print(f"Found {len(self.samples)} samples for {train_val_key}")
        np.random.shuffle(self.samples)

        # Compute class weights
        if compute_weights and self.train_val_key == "train":
            self.class_counts = class_counts

            # Median frequency balancing
            nonzero = class_counts[class_counts > 0]
            median = np.median(nonzero)

            weights = median / (class_counts + 1e-6)

            # Optional: avoid extreme rare-class weights
            weights = np.clip(weights, 0.0, 10.0)

            self.class_weights = torch.tensor(weights, dtype=torch.float32)

            # print("Class counts:", self.class_counts)
            # print("Class weights:", self.class_weights)
        else:
            self.class_counts = None
            self.class_weights = None

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
        self.s2_tiles = s2_tiles # "T11SMT"
        self.labels_path = os.path.join(top_dir, labels, self.s2_tiles)
        # self.metadata_path = metadata_path
        self.train_val_key = train_val_key
        # self.labels = []

        if self.s2_tiles == "T11SMT":
            if self.train_val_key == "train":
                img_label_pairs = {
                    '20180603T182919_mask_10m.tif': '20180603T182919.tif',
                    '20180906T182911_mask_10m.tif': '20180906T182911.tif',
                    '20191021T183411_mask_10m.tif': '20191021T183411.tif',
                    '20200712T182919_mask_10m.tif': '20200712T182919.tif',
                    '20200905T182921_mask_10m.tif': '20200905T182921.tif',
                    '20201015T183341_mask_10m.tif': '20201015T183341.tif',
                    '20201104T183541_mask_10m.tif': '20201104T183541.tif',
                }

            elif self.train_val_key == "val":
                img_label_pairs = {
                    '20191105T183539_mask_10m.tif': '20191105T183539.tif',  # Validation
                    '20210905T182919_mask_10m.tif': '20210905T182919.tif',  # Validation
                }
        elif self.s2_tiles == "T16REV":
            if self.train_val_key == "train":
                img_label_pairs = {
                    '20180405T161859_mask_10m.tif': '20180405T161859.tif',
                    '20190321T161949_mask_10m.tif': '20190321T161949.tif',
                    '20190410T161839_mask_10m.tif': '20190410T161839.tif',
                    '20190614T161901_mask_10m.tif': '20190614T161901.tif',
                    '20200409T161901_mask_10m.tif': '20200409T161901.tif',
                    '20211110T162521_mask_10m.tif': '20211110T162521.tif',
                }

            elif self.train_val_key == "val":
                img_label_pairs = {
                    '20180510T161901_mask_10m.tif': '20180510T161901.tif',  # Validation
                    '20190515T161901_mask_10m.tif': '20190515T161901.tif',  # Validation
                    '20200229T162211_mask_10m.tif': '20200229T162211.tif',  # Validation
                    '20210509T161829_mask_10m.tif': '20210509T161829.tif',  # Validation
                }
        else:
            raise ValueError(f"Unknown tile {self.s2_tiles}")


        t0 = datetime(2015, 1, 1)
        xy_offset_points = [i for i in range(0, 1536, 128)]
        self.samples = []
        for t in tqdm(img_label_pairs.keys(), desc=f"Building {train_val_key} image label pairs"):
            # read the label with corresponding 
            tile_containing_img = img_label_pairs[t]
            tile_ds = rasterio.open(os.path.join(self.top_dir, self.s2_tiles ,tile_containing_img))
             
            date = tile_containing_img.split(".tif")[0]
            dt = datetime.strptime(date , "%Y%m%dT%H%M%S")   
            doy = (dt - t0).total_seconds() / 86400.0  # days since t0
            
            label_ds = rasterio.open(os.path.join(self.labels_path, t))
            # iterate through xy_offset_points
            for x in xy_offset_points:
                for y in xy_offset_points:
                    #read labels with this x and y as the top left corner, and 128x128 window size
                    label_window = Window(x, y, 128, 128)
                    label_patch = label_ds.read(window=label_window)
                    left, bottom, right, top = window_bounds(label_window, transform=label_ds.transform)
                    s2_window = from_bounds(left, bottom, right, top, transform=tile_ds.transform)
                    # read the same area in image
                    s2_patch = tile_ds.read(window=s2_window)
                    assert label_patch.shape[1] == 128 and label_patch.shape[2] == 128, f"S2 patch shape {s2_patch.shape} does not match label patch shape {label_patch.shape}"
                    assert s2_patch.shape[1] == 128 and s2_patch.shape[2] == 128, f"S2 patch shape {s2_patch.shape} does not match label patch shape {label_patch.shape}"
                    label_patch[label_patch == -1] = 0
                    assert set(np.unique(label_patch)).issubset({0, 1}), \
                        f"Invalid label values: {np.unique(label_patch)}"
                    if label_patch.sum() == 0: # do not use labels with only non-burned pixels
                        continue
                    else:
                        #fill -1 values in label with 0 (non-burned)
                        burned_pixel_count = (label_patch > 0).sum()
                        self.samples.append({
                            "doy": doy,
                            "x": int(s2_window.col_off),
                            "y": int(s2_window.row_off),
                            "s2_img_patch": s2_patch,
                            "label": label_patch.squeeze(),  # (128,128) uint8 with values {0.0, 1.0}
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
    
