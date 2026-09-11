#!/usr/bin/env python3
"""Validate downloaded LIANet Tessera/AlphaEarth embedding GeoTIFFs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def require_deps():
    missing = []
    try:
        import rasterio  # noqa: F401
    except ImportError:
        missing.append("rasterio")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        missing.append("pyarrow")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    if missing:
        raise SystemExit(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Run this in the embedding/TerraTorch environment."
        )


def as_tuple(value: Any) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "element" in parsed[0]:
            return tuple(p["element"] for p in parsed)
        return tuple(parsed)
    return tuple(value)


def expected_path(row: dict, embedding_dir: Path | None, kind: str) -> Path | None:
    p = row.get("embedding_path")
    if isinstance(p, str) and p.strip():
        path = Path(p)
        if embedding_dir is not None and not path.is_absolute():
            return embedding_dir / path
        return path
    if embedding_dir is None:
        return None
    suffix = "tessera" if kind == "tessera" else "aef"
    return embedding_dir / f"{row['sample_id']}_{suffix}.tif"


def transform_close(actual, expected_gt, tol: float) -> bool:
    actual_gt = (actual.c, actual.a, actual.b, actual.f, actual.d, actual.e)
    return all(abs(float(a) - float(e)) <= tol for a, e in zip(actual_gt, expected_gt))


def validate_row(row: dict, path: Path | None, kind: str, sample_pixels: int, transform_tol: float) -> dict:
    import numpy as np
    import rasterio

    expected_bands = 128 if kind == "tessera" else 64
    result = {
        "sample_id": row.get("sample_id", ""),
        "kind": kind,
        "path": "" if path is None else str(path),
        "ok": True,
        "errors": "",
        "file_size_bytes": 0,
        "bands": None,
        "height": None,
        "width": None,
        "dtype": "",
        "nan_fraction": None,
        "zero_fraction": None,
        "constant_sample_bands": None,
        "compression": "",
        "tiled": None,
        "block_shape": "",
    }
    errors = []

    if path is None:
        errors.append("missing_path")
        result["ok"] = False
        result["errors"] = ";".join(errors)
        return result
    if not path.exists() or path.stat().st_size == 0:
        errors.append("file_missing_or_empty")
        result["ok"] = False
        result["errors"] = ";".join(errors)
        return result

    result["file_size_bytes"] = path.stat().st_size
    expected_shape = as_tuple(row["raster_shape"])
    expected_gt = as_tuple(row["geotransform"])

    with rasterio.open(path) as src:
        result["bands"] = src.count
        result["height"] = src.height
        result["width"] = src.width
        result["dtype"] = ",".join(src.dtypes)
        result["compression"] = str(src.compression.value if src.compression else "")
        result["tiled"] = bool(src.is_tiled)
        result["block_shape"] = str(src.block_shapes[0] if src.block_shapes else "")

        if src.count != expected_bands:
            errors.append(f"band_count:{src.count}!={expected_bands}")
        if (src.height, src.width) != (int(expected_shape[0]), int(expected_shape[1])):
            errors.append(f"shape:{(src.height, src.width)}!={expected_shape}")
        if src.crs is None or str(src.crs) != str(row["crs"]):
            errors.append("crs_mismatch")
        if not transform_close(src.transform, expected_gt, transform_tol):
            errors.append("transform_mismatch")
        if any(dtype != "float32" for dtype in src.dtypes):
            errors.append(f"dtype:{src.dtypes}")
        if not src.is_tiled:
            errors.append("not_tiled")
        if src.compression is None:
            errors.append("not_compressed")

        read_height = min(src.height, sample_pixels)
        read_width = min(src.width, sample_pixels)
        arr = src.read(window=((0, read_height), (0, read_width))).astype("float32")
        finite = np.isfinite(arr)
        result["nan_fraction"] = float(1.0 - finite.mean())
        result["zero_fraction"] = float((arr == 0).mean())

        if not finite.any():
            errors.append("all_nan_sample")
        if np.all(arr == 0):
            errors.append("all_zero_sample")
        constant_bands = 0
        for band in arr:
            finite_band = band[np.isfinite(band)]
            if finite_band.size and float(finite_band.min()) == float(finite_band.max()):
                constant_bands += 1
        result["constant_sample_bands"] = constant_bands
        if constant_bands == src.count:
            errors.append("all_sample_bands_constant")

    result["ok"] = not errors
    result["errors"] = ";".join(errors)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict]) -> None:
    ok = sum(1 for row in rows if row["ok"])
    failed = len(rows) - ok
    lines = [
        "# Embedding Validation Report",
        "",
        f"- Rows checked: {len(rows)}",
        f"- Passed: {ok}",
        f"- Failed: {failed}",
        "",
    ]
    if failed:
        lines.append("## Failures")
        lines.append("")
        for row in rows:
            if not row["ok"]:
                lines.append(f"- `{row['sample_id']}`: {row['errors']} ({row['path']})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--kind", required=True, choices=["tessera", "alphaearth"])
    parser.add_argument("--embedding-dir", default=None)
    parser.add_argument("--out-dir", default="/dccstor/geofm-datasets/datasets/lianet_bench/embeddings/outputs/qa")
    parser.add_argument("--sample-pixels", type=int, default=128)
    parser.add_argument("--transform-tol", type=float, default=1e-6)
    args = parser.parse_args()

    require_deps()

    import pyarrow.parquet as pq

    manifest_path = Path(args.manifest)
    embedding_dir = Path(args.embedding_dir) if args.embedding_dir else None
    out_dir = Path(args.out_dir)

    rows = pq.read_table(manifest_path).to_pylist()
    results = [
        validate_row(
            row=row,
            path=expected_path(row, embedding_dir, args.kind),
            kind=args.kind,
            sample_pixels=args.sample_pixels,
            transform_tol=args.transform_tol,
        )
        for row in rows
    ]

    stem = f"{manifest_path.stem}_{args.kind}"
    write_csv(out_dir / f"{stem}_validation.csv", results)
    write_report(out_dir / f"{stem}_validation.md", results)

    failures = sum(1 for row in results if not row["ok"])
    print(f"Checked {len(results)} rows; failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
