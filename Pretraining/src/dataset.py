import os
import glob
from abc import abstractmethod
import numpy as np
import rasterio as rio
import torch
from torch.utils.data import Dataset
from datetime import datetime
from utils import _preprocess_S2
import math

class LIANetPretrainingDataset(Dataset):
    def __init__(self, iterations_per_epoch, topdir_dataset, image_size, 
                 complete_tile_size, tile_names):
        self.topdir_dataset = topdir_dataset
        assert os.path.isdir(topdir_dataset)
        self.iterations_per_epoch = iterations_per_epoch
        self.image_size = image_size
        self.complete_tile_size = complete_tile_size
        self.tile_names_list = tile_names
        self.tiles = {}
        for tile_name in self.tile_names_list:
            tile_dir = os.path.join(self.topdir_dataset, tile_name)
            tif_files = os.listdir(tile_dir)
            self.tiles[tile_name] = tif_files

    @property
    def _epoch(self) -> int:
        return self._epoch_shared.value

    @abstractmethod
    def _rand_xy(self):
        x0 = np.random.choice(np.arange(0, self.complete_tile_size , dtype=int))
        y0 = np.random.choice(np.arange(0, self.complete_tile_size , dtype=int))
        return x0,y0
    
    @abstractmethod
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

    @abstractmethod
    def __getitem__(self, i):

        # select a random tile index
        tile_idx = np.random.randint(0, len(self.tile_names_list)) #TODO: add weights based on number of files per tile
        tile_name = self.tile_names_list[tile_idx]
        # excluding 2024 from training set and selecting a random time index from the rest 
        valid_tiles = [f for f in self.tiles[tile_name] 
            if f.endswith(".tif") and not f.startswith("2024")
        ]
        time_idx = np.random.randint(0, len(valid_tiles))
        dt_properties = self._get_dt_properties(time_idx, valid_tiles)

        # select a random spatial location
        x0, y0 = self._rand_xy()
        # Use it in image space
        x0_img = 10980 - self.image_size if x0 > 10980 - self.image_size else x0
        y0_img = 10980 - self.image_size if y0 > 10980 - self.image_size else y0
        window = rio.windows.Window(col_off=x0_img, row_off=y0_img,
                                    width=self.image_size, height=self.image_size)
        s2_path = glob.glob(os.path.join(self.topdir_dataset, tile_name ,f"{dt_properties['file_name']}"))
        assert len(s2_path) == 1
        s2_path = s2_path[0]
        with rio.open(s2_path) as src:
            patch = src.read(window=window)
        patch = _preprocess_S2(patch)

        # USe random pints in latent space
        x0_latent = (tile_idx) * 10980   + x0 - 984 # 984 is the overlap area of two tiles in x direction
        y0_latent = y0
        if x0_latent > 20976-self.image_size: # The most right boarder condition
            x0_latent = 20976 - self.image_size  # the maximum it can have
        if y0_latent > 10980 - self.image_size: # The most bottom boarder condition
            y0_latent = 10980 - self.image_size  # the maximum it can have

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
        valid_tiles = [f for f in self.tiles[tile_name] 
            if f.endswith(".tif") and f.startswith("2024")
        ]
        time_idx = np.random.randint(0, len(valid_tiles))
        dt_properties = self._get_dt_properties(time_idx, valid_tiles)

        # select a random spatial location
        x0, y0 = self._rand_xy()
        # Use it in image space
        x0_img = 10980 - self.image_size if x0 > 10980 - self.image_size else x0
        y0_img = 10980 - self.image_size if y0 > 10980 - self.image_size else y0
        window = rio.windows.Window(col_off=x0_img, row_off=y0_img,
                                    width=self.image_size, height=self.image_size)
        s2_path = glob.glob(os.path.join(self.topdir_dataset, tile_name ,f"{dt_properties['file_name']}"))
        assert len(s2_path) == 1
        s2_path = s2_path[0]
        with rio.open(s2_path) as src:
            patch = src.read(window=window)
        patch = _preprocess_S2(patch)

        # USe random pints in latent space
        x0_latent = (tile_idx) * 10980   + x0 - 984 # 984 is the overlap area of two tiles in x direction
        y0_latent = y0
        if x0_latent > 20976-self.image_size: # The most right boarder condition
            x0_latent = 20976 - self.image_size  # the maximum it can have
        if y0_latent > 10980 - self.image_size: # The most bottom boarder condition
            y0_latent = 10980 - self.image_size  # the maximum it can have


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

if __name__ == "__main__":
    dataset = LIANetPretrainingPlotterDataset(iterations_per_epoch=1000,
                             topdir_dataset="/home/user/data/extended_area_experiment_data",
                             image_size=128,
                             complete_tile_size=10980,
                             tile_names=["T32UPU", "T32UQU"])
    i = dataset[0]
    print(i['tile_name'])