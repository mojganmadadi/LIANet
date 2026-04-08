import os
import glob
import math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio as rio
from rasterio.warp import transform_bounds

from utils import _preprocess_S2


class LIANetFixedGridTimeEvalDataset(Dataset):
    """
    Deterministic evaluation dataset over:
      - uniform spatial grid in latent mosaic coords (x_s2, y_s2)
      - uniform query times across the covered period (query_ts)
    For each sample, picks the nearest available acquisition in abs time for the chosen tile.

    Returns dict compatible with your model inputs:
      - delta_days, doy_sin, doy_cos (time embedding inputs)
      - x_s2, y_s2 (latent coords)
      - x_s2_img, y_s2_img (tile coords)
      - tile_name, date_str
      - query_ts, chosen_ts, abs_dt_days (diagnostics)
      - s2data (preprocessed patch)
    """

    def __init__(
        self,
        topdir_dataset: str,
        image_size: int,
        tile_names: list[str],
        complete_tile_size: int = 10980,
        single_tile_width: int = 10980,
        num_x: int = 32,
        num_y: int = 32,
        num_t: int = 24,
        t_start: datetime | None = None,
        t_end: datetime | None = None,
        overlap_mode: str = "left",  # "left" | "right" | "both"
        include_2024: bool = True,
        file_glob: str = "*.tif",
        skip_bf: bool = True,
        deterministic_order: bool = True,
    ):
        super().__init__()
        self.topdir_dataset = Path(topdir_dataset)
        assert self.topdir_dataset.is_dir(), f"Not a dir: {topdir_dataset}"

        self.image_size = int(image_size)
        self.complete_tile_size = int(complete_tile_size)
        self.single_tile_width = int(single_tile_width)
        self.tile_names_list = list(tile_names)
        assert len(self.tile_names_list) >= 1

        assert overlap_mode in ("left", "right", "both")
        self.overlap_mode = overlap_mode

        # -------------------------
        # Collect files + timestamps per tile
        # -------------------------
        self.tile_files: dict[str, list[str]] = {}
        self.tile_times: dict[str, np.ndarray] = {}

        for tile in self.tile_names_list:
            tile_dir = self.topdir_dataset / tile
            assert tile_dir.is_dir(), f"Missing tile dir: {tile_dir}"

            files = [p.name for p in tile_dir.glob(file_glob)]
            if skip_bf:
                files = [f for f in files if "bf" not in f.lower()]

            if not include_2024:
                files = [f for f in files if not f.startswith("2024")]

            if len(files) == 0:
                raise ValueError(f"No tif files found for tile {tile} in {tile_dir}")

            # Parse timestamps from filename stem: YYYYmmddTHHMMSS
            times = []
            good_files = []
            for f in files:
                stem = Path(f).stem
                try:
                    dt = datetime.strptime(stem, "%Y%m%dT%H%M%S")
                except ValueError:
                    # ignore non-conforming
                    continue
                good_files.append(f)
                times.append(int(dt.timestamp()))

            if len(good_files) == 0:
                raise ValueError(f"No timestamp-parsable files in {tile_dir}")

            # Sort by time for stable nearest-neighbor
            order = np.argsort(np.asarray(times, dtype=np.int64))
            good_files = [good_files[i] for i in order]
            times = np.asarray([times[i] for i in order], dtype=np.int64)

            self.tile_files[tile] = good_files
            self.tile_times[tile] = times

        # -------------------------
        # Compute overlap_x and mosaic_width
        # -------------------------
        self.overlap_x = self._compute_overlap_x()
        self.mosaic_width = (
            self.single_tile_width * len(self.tile_names_list)
            - self.overlap_x * (len(self.tile_names_list) - 1)
        )
        self.mosaic_height = self.single_tile_width  # vertical is unchanged

        # -------------------------
        # Build uniform spatial grid (latent coords)
        # -------------------------
        # Valid top-left positions for a patch in latent coords
        x_min, x_max = 0, self.mosaic_width - self.image_size
        y_min, y_max = 0, self.mosaic_height - self.image_size
        if x_max < x_min or y_max < y_min:
            raise ValueError("image_size too large for mosaic/tile size")

        # Uniformly spaced top-left corners
        xs = np.linspace(x_min, x_max, num=num_x, dtype=np.int32)
        ys = np.linspace(y_min, y_max, num=num_y, dtype=np.int32)

        # -------------------------
        # Build uniform query times
        # -------------------------
        # Global min/max by default (union across tiles)
        all_times = np.concatenate([self.tile_times[t] for t in self.tile_names_list])
        global_min_ts = int(all_times.min())
        global_max_ts = int(all_times.max())

        if t_start is None:
            t0 = datetime.utcfromtimestamp(global_min_ts)
        else:
            t0 = t_start
        if t_end is None:
            t1 = datetime.utcfromtimestamp(global_max_ts)
        else:
            t1 = t_end
        if t1 <= t0:
            raise ValueError("t_end must be > t_start")

        # Uniform query timestamps (seconds)
        t0_ts = int(t0.replace(tzinfo=None).timestamp())
        t1_ts = int(t1.replace(tzinfo=None).timestamp())
        query_ts = np.linspace(t0_ts, t1_ts, num=num_t, dtype=np.int64)

        # -------------------------
        # Build sample index (deterministic)
        # -------------------------
        # We create samples in latent mosaic coords, then map to tile coords.
        # For overlap strip, you can request left/right/both.
        self.samples = []
        for t_q in query_ts:
            for y in ys:
                for x in xs:
                    tile_choices = self._latent_x_to_tile_choices(x)
                    if overlap_mode == "left":
                        tile_choices = tile_choices[:1]
                    elif overlap_mode == "right":
                        tile_choices = tile_choices[-1:]
                    elif overlap_mode == "both":
                        pass

                    for tile_idx in tile_choices:
                        tile_name = self.tile_names_list[tile_idx]
                        x_offset = tile_idx * (self.single_tile_width - self.overlap_x)
                        x_img = int(x - x_offset)
                        y_img = int(y)

                        # Safety clamp (should already be valid)
                        x_img = max(0, min(x_img, self.single_tile_width - self.image_size))
                        y_img = max(0, min(y_img, self.single_tile_width - self.image_size))

                        self.samples.append(
                            {
                                "tile_idx": tile_idx,
                                "tile_name": tile_name,
                                "x_s2": int(x),
                                "y_s2": int(y),
                                "x_s2_img": x_img,
                                "y_s2_img": y_img,
                                "query_ts": int(t_q),
                            }
                        )

        if deterministic_order:
            # already deterministic by construction; keep hook in case you later change construction
            pass

    # -------------------------
    # Geometry helpers
    # -------------------------
    def _compute_overlap_x(self) -> int:
        if len(self.tile_names_list) < 2:
            return 0

        # Use first timestamp file from each of first two tiles
        sample_paths = []
        for tile in self.tile_names_list[:2]:
            f0 = self.tile_files[tile][0]
            sample_paths.append(str(self.topdir_dataset / tile / f0))

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

    def _latent_x_to_tile_choices(self, x_latent: int) -> list[int]:
        """
        Map a latent x to candidate tiles that cover it.
        For 2 tiles:
          tile0 covers [0, 10980)
          tile1 covers [shift, shift+10980) where shift=(10980-overlap_x)
        Overlap region yields 2 candidates.
        """
        shift = self.single_tile_width - self.overlap_x
        candidates = []
        for tile_idx in range(len(self.tile_names_list)):
            left = tile_idx * shift
            right = left + self.single_tile_width
            if left <= x_latent < right:
                candidates.append(tile_idx)

        # Fallback (shouldn’t happen if x is within mosaic_width)
        if len(candidates) == 0:
            # snap to nearest tile
            tile_idx = int(round(x_latent / shift))
            tile_idx = max(0, min(tile_idx, len(self.tile_names_list) - 1))
            candidates = [tile_idx]
        return candidates

    # -------------------------
    # Time embedding helpers
    # -------------------------
    def _time_features_from_ts(self, ts: int):
        dt = datetime.utcfromtimestamp(int(ts))

        t0 = datetime(2015, 1, 1)
        delta_days = (dt - t0).total_seconds() / 86400.0

        doy = dt.timetuple().tm_yday
        doy_norm = (doy - 1) / 365.0
        doy_sin = math.sin(2 * math.pi * doy_norm)
        doy_cos = math.cos(2 * math.pi * doy_norm)
        return delta_days, doy_sin, doy_cos, dt

    def _nearest_time_index(self, tile_name: str, query_ts: int) -> int:
        times = self.tile_times[tile_name]  # sorted
        # argmin on abs diff
        return int(np.argmin(np.abs(times - int(query_ts))))

    # -------------------------
    # Torch Dataset API
    # -------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        tile_name = s["tile_name"]
        query_ts = int(s["query_ts"])

        # pick nearest available acquisition for THIS tile
        time_idx = self._nearest_time_index(tile_name, query_ts)
        chosen_ts = int(self.tile_times[tile_name][time_idx])
        date_str = self.tile_files[tile_name][time_idx]  # filename

        # time features (based on query OR chosen?)
        # For evaluation of temporal generalization, it’s usually cleaner to feed the QUERY time
        # (what you intended), while the image comes from the nearest acquisition.
        delta_days, doy_sin, doy_cos, query_dt = self._time_features_from_ts(query_ts)

        abs_dt_days = abs(chosen_ts - query_ts) / 86400.0

        # read patch
        x_img = int(s["x_s2_img"])
        y_img = int(s["y_s2_img"])
        window = rio.windows.Window(col_off=x_img, row_off=y_img,
                                    width=self.image_size, height=self.image_size)

        tif_path = str(self.topdir_dataset / tile_name / date_str)
        with rio.open(tif_path) as src:
            patch = src.read(window=window)

        patch = _preprocess_S2(patch)

        return {
            # time embedding inputs (feed these to your learned Fourier time encoder)
            "delta_days": torch.tensor(delta_days, dtype=torch.float32),
            "doy_sin": torch.tensor(doy_sin, dtype=torch.float32),
            "doy_cos": torch.tensor(doy_cos, dtype=torch.float32),

            # diagnostics
            "query_ts": torch.tensor(query_ts, dtype=torch.int64),
            "chosen_ts": torch.tensor(chosen_ts, dtype=torch.int64),
            "abs_dt_days": torch.tensor(abs_dt_days, dtype=torch.float32),
            "time_idx": torch.tensor(time_idx, dtype=torch.int32),
            "date_str": date_str,

            # spatial coords
            "x_s2": int(s["x_s2"]),
            "y_s2": int(s["y_s2"]),
            "x_s2_img": x_img,
            "y_s2_img": y_img,
            "tile_name": tile_name,

            # data
            "s2data": patch,
        }


if __name__ == "__main__":
    ds = LIANetFixedGridTimeEvalDataset(
        topdir_dataset="/home/user/data_shared",
        image_size=128,
        tile_names=["T16TEK", "T16TFK"],
        num_x=24,
        num_y=24,
        num_t=32,
        overlap_mode="both",   # evaluate overlap twice (left+right)
        include_2024=True,
    )
    print("len:", len(ds))
    ex = ds[0]
    print(ex["tile_name"], ex["date_str"], ex["abs_dt_days"].item(), ex["x_s2"], ex["y_s2"])
    print("mosaic_width:", ds.mosaic_width, "overlap_x:", ds.overlap_x)
