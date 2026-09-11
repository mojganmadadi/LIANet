#!/usr/bin/env python3
"""Download yearly AlphaEarth embeddings from a LIANet manifest.

This wrapper patches the inspected AlphaEarth helper so it reads all 64 bands
and reuses the already-loaded index table.
"""

from __future__ import annotations

import argparse
import os
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_TERRATORCH_ROOT = os.environ.get("TERRATORCH_ROOT", "")


def add_terratorch_path(path: str) -> None:
    if not path:
        return
    p = Path(path)
    if p.exists():
        sys.path.insert(0, str(p))


def patch_alphaearth(module, block_size: int) -> None:
    import numpy as np
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio
    from rasterio.env import Env
    from rasterio.enums import Resampling
    from rasterio.windows import Window, from_bounds
    from rasterio.warp import reproject, transform_bounds

    class AlphaEarthNoOverlapError(RuntimeError):
        pass

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

    def window_from_bbox_wgs84(
        *,
        bbox_wgs84,
        src_crs,
        src_transform,
        src_width,
        src_height,
        pad_pixels=16,
    ):
        transformed_bbox = transform_bounds(
            "EPSG:4326", src_crs, *bbox_wgs84, densify_pts=21
        )
        left, bottom, right, top = transformed_bbox
        w = from_bounds(left, bottom, right, top, transform=src_transform)

        col_off = math.floor(w.col_off) - pad_pixels
        row_off = math.floor(w.row_off) - pad_pixels
        width = math.ceil(w.width) + 2 * pad_pixels
        height = math.ceil(w.height) + 2 * pad_pixels

        col_start = max(0, col_off)
        row_start = max(0, row_off)
        col_stop = min(src_width, col_off + width)
        row_stop = min(src_height, row_off + height)

        if col_start >= col_stop or row_start >= row_stop:
            raise AlphaEarthNoOverlapError(
                "Empty/non-overlapping AlphaEarth window "
                f"for bbox={bbox_wgs84}, transformed_bbox={transformed_bbox}, "
                f"window=({col_start}, {row_start}, {col_stop}, {row_stop}), "
                f"src_shape=({src_height}, {src_width})"
            )

        return Window(
            col_start,
            row_start,
            col_stop - col_start,
            row_stop - row_start,
        )

    def fetch_alphaearth_tile(tile_info, bbox_wgs84=None):
        file_path = tile_info["location"]
        path = file_path.split("aef/", 1)[1]
        url = f"https://data.source.coop/tge-labs/aef/{path[:-5]}.vrt"

        with Env(
            AWS_NO_SIGN_REQUEST="YES",
            AWS_REGION="us-west-2",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            VSI_CACHE="TRUE",
            VSI_CACHE_SIZE="536870912",
            CPL_VSIL_CURL_CACHE_SIZE="536870912",
            GDAL_HTTP_MAX_RETRY="5",
            GDAL_HTTP_RETRY_DELAY="1",
        ):
            with rasterio.open(f"/vsicurl/{url}") as src:
                if bbox_wgs84 is not None:
                    win = window_from_bbox_wgs84(
                        bbox_wgs84=bbox_wgs84,
                        src_crs=src.crs.to_string(),
                        src_transform=src.transform,
                        src_width=src.width,
                        src_height=src.height,
                        pad_pixels=16,
                    )
                    data = src.read(window=win)
                    win_transform = rasterio.windows.transform(win, src.transform)
                else:
                    data = src.read()
                    win_transform = src.transform

                if data.shape[0] != 64:
                    raise RuntimeError(f"Expected 64 AlphaEarth bands, got {data.shape[0]} from {url}")
                return module.dequantize_alphaearth(data), str(src.crs), win_transform

    def fetch_alphaearth_to_grid(
        *,
        year,
        bbox_wgs84,
        dst_crs,
        dst_transform,
        dst_height,
        dst_width,
        out_path,
        index_table=None,
        resampling=Resampling.nearest,
    ):
        if index_table is None:
            index_table = module.load_alphaearth_index()
        tiles = module.find_alphaearth_tiles(index_table, bbox_wgs84, year)
        if not tiles:
            raise RuntimeError(f"No AlphaEarth tiles found for bbox={bbox_wgs84}, year={year}")

        dst_arr = np.full((64, dst_height, dst_width), np.nan, dtype=np.float32)
        skipped_tiles = []
        filled_any = False
        for tile_info in tiles:
            try:
                emb_chw, src_crs, src_transform = fetch_alphaearth_tile(tile_info, bbox_wgs84=bbox_wgs84)
            except AlphaEarthNoOverlapError as exc:
                skipped_tiles.append(str(exc))
                print(f"Skipping non-overlapping AlphaEarth candidate: {exc}")
                continue

            dst = np.full((64, dst_height, dst_width), np.nan, dtype=np.float32)
            reproject(
                source=emb_chw,
                destination=dst,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            contributed = ~np.isnan(dst[0])
            empty = np.isnan(dst_arr[0])
            fill_here = contributed & empty
            if np.any(fill_here):
                dst_arr[:, fill_here] = dst[:, fill_here]
                filled_any = True

        if not filled_any:
            detail = f" Skipped candidates: {'; '.join(skipped_tiles[:3])}" if skipped_tiles else ""
            raise RuntimeError(
                f"No AlphaEarth pixels overlapped bbox={bbox_wgs84}, year={year} "
                f"after trying {len(tiles)} candidate tiles.{detail}"
            )

        write_geotiff(out_path, dst_arr, crs=dst_crs, transform=dst_transform)
        return out_path

    def get_alphaearth_from_manifest(manifest_path, out_dir, nworkers=8):
        manifest_path = Path(manifest_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        table, jobs = module.build_jobs_from_manifest(manifest_path, out_dir)
        if not jobs:
            print("No jobs found. Nothing to do.")
            return

        print("Loading AlphaEarth index...")
        index_table = module.load_alphaearth_index()
        print(f"Index loaded with {len(index_table)} tiles")

        updates = {}
        failures = []

        def run_job(job):
            dst_transform = module.affine_from_geotransform(job.geotransform)
            height, width = job.raster_shape
            bbox_src = module.bbox_from_affine(dst_transform, height, width)
            bbox_wgs84 = module.reproject_bbox(bbox_src, job.crs, "EPSG:4326")

            out_path = job.out_path
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"Skipping {job.sample_id} (already exists)")
                return job.idx, str(out_path)

            print(f"Processing {job.sample_id} (year={job.year})")
            fetch_alphaearth_to_grid(
                year=job.year,
                bbox_wgs84=bbox_wgs84,
                dst_crs=job.crs,
                dst_transform=dst_transform,
                dst_height=height,
                dst_width=width,
                out_path=out_path,
                index_table=index_table,
                resampling=Resampling.nearest,
            )
            return job.idx, str(out_path)

        with ThreadPoolExecutor(max_workers=min(nworkers, len(jobs))) as ex:
            futures = {ex.submit(run_job, job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    idx, new_path = fut.result()
                except Exception as exc:
                    failures.append((job.idx, job.sample_id, str(exc)))
                    print(f"Failed {job.sample_id} (row={job.idx}, year={job.year}): {exc}")
                    continue
                updates[idx] = new_path

        rows = table.to_pylist()
        for idx, new_path in updates.items():
            rows[idx]["embedding_path"] = new_path
        new_table = pa.Table.from_pylist(rows)
        tmp_path = Path(str(manifest_path) + ".tmp")
        pq.write_table(new_table, tmp_path, compression="snappy")
        os.replace(tmp_path, manifest_path)
        print(f"Updated {len(updates)} rows in manifest: {manifest_path}")
        if failures:
            print(f"AlphaEarth failures: {len(failures)} rows were left unchanged in {manifest_path}")
            for idx, sample_id, message in failures[:20]:
                print(f"  row={idx} sample={sample_id}: {message}")
            if len(failures) > 20:
                print(f"  ... {len(failures) - 20} more failures omitted")

    module.write_geotiff = write_geotiff
    module.fetch_alphaearth_tile = fetch_alphaearth_tile
    module.fetch_alphaearth_to_grid = fetch_alphaearth_to_grid
    module.get_alphaearth_from_manifest = get_alphaearth_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nworkers", type=int, default=2)
    parser.add_argument("--terratorch-root", default=DEFAULT_TERRATORCH_ROOT)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    add_terratorch_path(args.terratorch_root)

    from terratorch.tasks import alpha_earth as alphaearth_module

    patch_alphaearth(alphaearth_module, args.block_size)
    alphaearth_module.get_alphaearth_from_manifest(
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out_dir),
        nworkers=args.nworkers,
    )


if __name__ == "__main__":
    main()
