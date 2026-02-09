import os
import glob
from abc import abstractmethod
import numpy as np
import rasterio as rio
from rasterio.warp import transform_bounds
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from utils import _preprocess_S2
import math


class LIANetPretrainingDataset(Dataset):
    def __init__(self, iterations_per_epoch, topdir_dataset, image_size, 
                 complete_tile_size, tile_names, time_sampling="available"):
        self.topdir_dataset = topdir_dataset
        assert os.path.isdir(topdir_dataset)
        self.iterations_per_epoch = iterations_per_epoch
        self.image_size = image_size
        self.complete_tile_size = complete_tile_size
        self.tile_names_list = tile_names
        self.time_sampling = time_sampling
        self.tiles = {}
        self.tile_files = {}
        self.tile_times = {}
        self.tile_years = {}
        for tile_name in self.tile_names_list:
            tile_dir = os.path.join(self.topdir_dataset, tile_name)
            tif_files = os.listdir(tile_dir)
            self.tiles[tile_name] = [f for f in tif_files if "bf" not in f.lower()]
            valid_files = [f for f in self.tiles[tile_name] if f.endswith(".tif")]
            self.tile_files[tile_name] = valid_files
            times = []
            for f in valid_files:
                capture_time = os.path.splitext(os.path.basename(f))[0]
                dt = datetime.strptime(capture_time, "%Y%m%dT%H%M%S")
                times.append(int(dt.timestamp()))
            self.tile_times[tile_name] = np.asarray(times, dtype=np.int64)
            self.tile_years[tile_name] = np.asarray(
                [datetime.utcfromtimestamp(t).year for t in self.tile_times[tile_name]],
                dtype=np.int16
            )

        self.single_tile_width = 10980
        self.overlap_x = self._compute_overlap_x()
        self.mosaic_width = (
            self.single_tile_width * len(self.tile_names_list)
            - self.overlap_x * (len(self.tile_names_list) - 1)
        )

    def _compute_overlap_x(self) -> int:
        if len(self.tile_names_list) < 2:
            return 0

        # Use the first valid .tif from each tile to compute horizontal overlap.
        sample_paths = []
        for tile_name in self.tile_names_list[:2]:
            candidates = [
                f for f in self.tiles[tile_name]
                if f.endswith(".tif") and "bf" not in f.lower()
            ]
            if not candidates:
                raise ValueError(f"No valid .tif files found for tile {tile_name}")
            sample_paths.append(os.path.join(self.topdir_dataset, tile_name, candidates[0]))

        with rio.open(sample_paths[0]) as s1, rio.open(sample_paths[1]) as s2:
            ref_crs = s1.crs
            if ref_crs is None or s2.crs is None:
                raise ValueError("Missing CRS on one or more tiles; cannot compute overlap.")

            b1 = s1.bounds
            b2 = s2.bounds
            if s2.crs != ref_crs:
                b2 = transform_bounds(s2.crs, ref_crs, *b2)

            overlap_m = max(0.0, min(b1.right, b2.right) - max(b1.left, b2.left))
            pixel_size_x = abs(s1.transform.a)
            if pixel_size_x <= 0:
                raise ValueError("Invalid pixel size; cannot compute overlap.")

            return int(round(overlap_m / pixel_size_x))

    def _rand_xy(self):
        x0 = np.random.choice(np.arange(0, self.complete_tile_size , dtype=int))
        y0 = np.random.choice(np.arange(0, self.complete_tile_size , dtype=int))
        return x0,y0
    
    def _get_dt_properties(self, time_idx, valid_tiles):

        s2_file_name = valid_tiles[time_idx] # for now I am using everything in the train

        capture_time = os.path.splitext(os.path.basename(s2_file_name))[0]
        dt = datetime.strptime(capture_time, "%Y%m%dT%H%M%S")

        t0 = datetime(2015, 1, 1)
        delta = (dt - t0).total_seconds() / 86400.0  # days since t0

        # day-of-year
        doy = dt.timetuple().tm_yday  # 1..365/366
        doy_norm = (doy - 1) / 365.0
        doy_sin = math.sin(2 * math.pi * doy_norm)
        doy_cos = math.cos(2 * math.pi * doy_norm)

        return {"file_name": s2_file_name,"delta_days": delta, "doy_sin": doy_sin, "doy_cos": doy_cos,}

    def _select_time_idx(self, tile_name):
        valid_tiles = self.tile_files[tile_name]
        if self.time_sampling == "available":
            return np.random.randint(0, len(valid_tiles))
        if self.time_sampling == "random_doy":
            times = self.tile_times[tile_name]
            year = int(np.random.choice(self.tile_years[tile_name]))
            doy = int(np.random.randint(1, 366))
            target_dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
            target_ts = int(target_dt.timestamp())
            return int(np.argmin(np.abs(times - target_ts)))
        raise ValueError(f"Unknown time_sampling '{self.time_sampling}'")

    def __getitem__(self, i):

        # select a random tile index
        tile_idx = np.random.randint(0, len(self.tile_names_list)) #TODO: add weights based on number of files per tile
        tile_name = self.tile_names_list[tile_idx]
        # excluding 2024 from training set and selecting a random time index from the rest 
        # valid_tiles = [f for f in self.tiles[tile_name] 
        #     if f.endswith(".tif") and not f.startswith("2024")
        # ]
        valid_tiles = self.tile_files[tile_name]
        time_idx = self._select_time_idx(tile_name)
        dt_properties = self._get_dt_properties(time_idx, valid_tiles)

        # select a random spatial location
        x0, y0 = self._rand_xy()
        # Use it in image space
        x0_img = self.single_tile_width - self.image_size if x0 > self.single_tile_width - self.image_size else x0
        y0_img = self.single_tile_width - self.image_size if y0 > self.single_tile_width - self.image_size else y0
        window = rio.windows.Window(col_off=x0_img, row_off=y0_img,
                                    width=self.image_size, height=self.image_size)
        s2_path = glob.glob(os.path.join(self.topdir_dataset, tile_name ,f"{dt_properties['file_name']}"))
        assert len(s2_path) == 1
        s2_path = s2_path[0]
        with rio.open(s2_path) as src:
            patch = src.read(window=window)
        patch = _preprocess_S2(patch)

        x_offset = tile_idx * (self.single_tile_width - self.overlap_x)  # 0 for first tile, 9996 for second
        x0_latent = x_offset + x0_img
        y0_latent = y0_img

        # clamp so the whole patch stays inside mosaic
        x0_latent = min(x0_latent, self.mosaic_width - self.image_size)
        y0_latent = min(y0_latent, self.single_tile_width - self.image_size)

        return {
            "delta_days": torch.tensor(dt_properties["delta_days"], dtype=torch.float32),
            "time_idx": torch.tensor(time_idx, dtype=torch.int32),
            "doy_sin": torch.tensor(dt_properties["doy_sin"], dtype=torch.float32),
            "doy_cos": torch.tensor(dt_properties["doy_cos"], dtype=torch.float32),
            "x_s2": x0_latent,
            "y_s2": y0_latent,
            "x_s2_img": x0_img,
            "y_s2_img": y0_img,
            "tile_name": tile_name,
            "s2data": patch
        }

    def __len__(self): return self.iterations_per_epoch


class LIANetPretrainingPlotterDataset(LIANetPretrainingDataset):

    def __getitem__(self, i):

        # select a random tile index
        tile_idx = np.random.randint(0, len(self.tile_names_list)) #TODO: add weights based on number of files per tile
        tile_name = self.tile_names_list[tile_idx]
        # excluding 2024 from training set and selecting a random time index from the rest 
        # valid_tiles = [f for f in self.tiles[tile_name] 
        #     if f.endswith(".tif") and f.startswith("2024")
        # ]
        valid_tiles = self.tile_files[tile_name]
        time_idx = self._select_time_idx(tile_name)
        dt_properties = self._get_dt_properties(time_idx, valid_tiles)

        # select a random spatial location
        x0, y0 = self._rand_xy()
        # Use it in image space
        x0_img = self.single_tile_width - self.image_size if x0 > self.single_tile_width - self.image_size else x0
        y0_img = self.single_tile_width - self.image_size if y0 > self.single_tile_width - self.image_size else y0
        window = rio.windows.Window(col_off=x0_img, row_off=y0_img,
                                    width=self.image_size, height=self.image_size)
        s2_path = glob.glob(os.path.join(self.topdir_dataset, tile_name ,f"{dt_properties['file_name']}"))
        assert len(s2_path) == 1
        s2_path = s2_path[0]
        with rio.open(s2_path) as src:
            patch = src.read(window=window)
        patch = _preprocess_S2(patch)

        x_offset = tile_idx * (self.single_tile_width - self.overlap_x)  # 0 for first tile, 9996 for second
        x0_latent = x_offset + x0_img
        y0_latent = y0_img

        # clamp so the whole patch stays inside mosaic
        x0_latent = min(x0_latent, self.mosaic_width - self.image_size)
        y0_latent = min(y0_latent, self.single_tile_width - self.image_size)

        return {
            "delta_days": torch.tensor(dt_properties["delta_days"], dtype=torch.float32),
            "time_idx": torch.tensor(time_idx, dtype=torch.int32),
            "date_str": valid_tiles[time_idx],
            "doy_sin": torch.tensor(dt_properties["doy_sin"], dtype=torch.float32),
            "doy_cos": torch.tensor(dt_properties["doy_cos"], dtype=torch.float32),
            "x_s2": x0_latent,
            "y_s2": y0_latent,
            "x_s2_img": x0_img,
            "y_s2_img": y0_img,
            "tile_name": tile_name,
            "s2data": patch
        }

    def __len__(self): return self.iterations_per_epoch

if __name__ == "__main__":
    dataset = LIANetPretrainingDataset(iterations_per_epoch=1000,
                             topdir_dataset="/home/user/data_shared",
                             image_size=128,
                             complete_tile_size=10980,
                            tile_names=["T32ULU"]
)
    i = dataset[0]
    print(dataset.mosaic_width)
