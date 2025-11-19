import os
import glob
from abc import abstractmethod
import numpy as np
import rasterio as rio
from torch.utils.data import Dataset

from utils import _preprocess_S2


class LIANetPretrainingDataset(Dataset):
    def __init__(self, iterations_per_epoch, topdir_dataset, image_size, 
                 complete_tile_size, ):
        self.topdir_dataset = topdir_dataset
        assert os.path.isdir(topdir_dataset)
        self.iterations_per_epoch = iterations_per_epoch
        self.image_size = image_size
        self.complete_tile_size = complete_tile_size
        

    @property
    def _epoch(self) -> int:
        return self._epoch_shared.value

    @abstractmethod
    def _rand_xy(self):
        x0 = np.random.choice(np.arange(0, self.complete_tile_size - self.image_size, dtype=int))
        y0 = np.random.choice(np.arange(0, self.complete_tile_size - self.image_size, dtype=int))
        return x0,y0
    
    @abstractmethod
    def __getitem__(self, i):

        time = int(np.random.randint(0, 4))  # 4 temporal indices
        
        x0, y0 = self._rand_xy()
        window = rio.windows.Window(col_off=x0, row_off=y0,
                                    width=self.image_size, height=self.image_size)

        s2_path = glob.glob(os.path.join(self.topdir_dataset, f"s2_{time}_*.tif"))
        assert len(s2_path) == 1
        s2_path = s2_path[0]
        with rio.open(s2_path) as src:
            patch = src.read(window=window)

        patch = _preprocess_S2(patch)
        return {
            "timestamp": time, "x_s2": x0, "y_s2": y0, "s2data": patch
        }

    def __len__(self): return self.iterations_per_epoch


class LIANetPretrainingPlotterDataset(LIANetPretrainingDataset):

    def __getitem__(self, i):

        x0, y0 = self._rand_xy()

        window = rio.windows.Window(col_off=x0, row_off=y0,
                                    width=self.image_size, height=self.image_size)

        s2_path_0 = glob.glob(os.path.join(self.topdir_dataset, f"s2_0_*.tif"))
        s2_path_1 = glob.glob(os.path.join(self.topdir_dataset, f"s2_1_*.tif"))
        s2_path_2 = glob.glob(os.path.join(self.topdir_dataset, f"s2_2_*.tif"))
        s2_path_3 = glob.glob(os.path.join(self.topdir_dataset, f"s2_3_*.tif"))

        assert len(s2_path_0) == 1
        assert len(s2_path_1) == 1
        assert len(s2_path_2) == 1
        assert len(s2_path_3) == 1

        with rio.open(s2_path_0[0]) as src:
            patch_0 = src.read(window=window)
        with rio.open(s2_path_1[0]) as src:
            patch_1 = src.read(window=window)
        with rio.open(s2_path_2[0]) as src:
            patch_2 = src.read(window=window)
        with rio.open(s2_path_3[0]) as src:
            patch_3 = src.read(window=window)

 
        # preprocess the patch
        patch_0 = _preprocess_S2(patch_0)
        patch_1 = _preprocess_S2(patch_1)
        patch_2 = _preprocess_S2(patch_2)
        patch_3 = _preprocess_S2(patch_3)

        return {"x_s2": x0, 
                "y_s2": y0,
                "s2data": np.stack([patch_0,patch_1,patch_2,patch_3]),  # shape (4, C, H, W)
               }

    def __len__(self): return self.iterations_per_epoch
