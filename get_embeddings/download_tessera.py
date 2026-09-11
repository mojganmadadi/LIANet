#!/usr/bin/env python3
"""Download yearly Tessera embeddings from a LIANet manifest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import os
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_TERRATORCH_ROOT = os.environ.get("TERRATORCH_ROOT", "")


def add_terratorch_path(path: str) -> None:
    if not path:
        return
    p = Path(path)
    if p.exists():
        sys.path.insert(0, str(p))


def patch_write_geotiff(module, block_size: int) -> None:
    import numpy as np
    import rasterio

    def choose_block(size: int) -> int:
        if size <= 32:
            return 16
        target = min(block_size, size - 16)
        return max(16, (target // 16) * 16)


    def write_geotiff(out_path, arr, *, crs, transform):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected (C,H,W) or (H,W), got {arr.shape}")

        height = int(arr.shape[1])
        width = int(arr.shape[2])
        bx = choose_block(width)
        by = choose_block(height)

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": int(arr.shape[0]),
            "dtype": arr.dtype,
            "crs": crs,
            "transform": transform,
            "tiled": True,
            "blockxsize": bx,
            "blockysize": by,
            "compress": "deflate",
            "predictor": 2 if np.issubdtype(arr.dtype, np.floating) else 1,
            "BIGTIFF": "IF_SAFER",
        }

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)

    module.write_geotiff = write_geotiff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nworkers", type=int, default=2)
    parser.add_argument("--tmp-tiles-dir", default=None)
    parser.add_argument("--terratorch-root", default=DEFAULT_TERRATORCH_ROOT)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    add_terratorch_path(args.terratorch_root)

    from terratorch.tasks import tessera as tessera_module

    patch_write_geotiff(tessera_module, args.block_size)
    tessera_module.get_tessera_from_manifest(
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out_dir),
        nworkers=args.nworkers,
        tmp_tiles_dir=Path(args.tmp_tiles_dir) if args.tmp_tiles_dir else None,
    )


if __name__ == "__main__":
    main()
