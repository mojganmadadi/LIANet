#!/usr/bin/env python3
"""Build LIANet train/val window manifests for Tessera and AlphaEarth.

The generated Parquet manifests follow the columns consumed by the existing
terratorch.tasks.tessera and terratorch.tasks.alpha_earth download helpers.
Extra columns are included for LIANet lookup/debugging.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


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
    if missing:
        raise SystemExit(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Run this in the embedding/TerraTorch environment."
        )


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    data_split: str
    embedding_path: str
    label_path: str
    crs: str
    geotransform: tuple[float, float, float, float, float, float]
    raster_shape: tuple[int, int]
    time_start: int
    lon: float | None
    lat: float | None
    tile: str
    dataset: str
    image_timestamp: str
    embedding_year: int
    source_s2_path: str
    x: int
    y: int
    height: int
    width: int
    grid_id: str
    fold: int | None = None


@dataclass(frozen=True)
class LookupRow:
    dataset: str
    tile: str
    data_split: str
    image_timestamp: str
    embedding_year: int
    sample_id: str
    grid_id: str
    source_s2_path: str
    label_path: str
    x: int
    y: int
    height: int
    width: int
    fold: int | None = None


def geotransform_from_transform(transform):
    return (
        float(transform.c),
        float(transform.a),
        float(transform.b),
        float(transform.f),
        float(transform.d),
        float(transform.e),
    )


def year_from_timestamp(timestamp: str) -> int:
    return int(timestamp[:4])


def split_from_fold(fold: int | None, val_folds: set[int]) -> str:
    if fold is None:
        return "unknown"
    return "val" if fold in val_folds else "train"


def parse_pastis_folds(metadata_path: Path) -> dict[str, int]:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit(
            "geopandas is required for PASTIS metadata.geojson. "
            "Run this in the embedding/TerraTorch environment."
        ) from exc

    gdf = gpd.read_file(metadata_path)
    patch_col = "ID_PATCH"
    if "fold" in gdf.columns:
        fold_col = "fold"
    elif "Fold" in gdf.columns:
        fold_col = "Fold"
    else:
        raise ValueError(f"No fold/Fold column found in {metadata_path}")

    folds = {}
    for _, row in gdf.iterrows():
        folds[str(row[patch_col])] = int(row[fold_col])
    return folds


def merge_row(
    rows_by_key: dict[tuple, ManifestRow],
    row: ManifestRow,
) -> ManifestRow:
    key = (row.dataset, row.tile, row.embedding_year, row.grid_id)
    existing = rows_by_key.get(key)
    if existing is None:
        rows_by_key[key] = row
        return row

    if existing.data_split == row.data_split or row.data_split == "unknown":
        return existing
    if existing.data_split == "unknown":
        merged = ManifestRow(**{**asdict(existing), "data_split": row.data_split})
    else:
        merged = ManifestRow(**{**asdict(existing), "data_split": "mixed"})
    rows_by_key[key] = merged
    return merged


def iter_selected_tiles(root: Path, selected: set[str] | None) -> list[Path]:
    tiles = sorted(p for p in root.iterdir() if p.is_dir())
    if selected:
        tiles = [p for p in tiles if p.name in selected]
    return tiles


def build_pastis(
    data_root: Path,
    val_folds: set[int],
    selected_tiles: set[str] | None,
    limit: int | None,
) -> tuple[list[ManifestRow], list[LookupRow]]:
    import rasterio
    from rasterio.windows import Window

    pastis_root = data_root / "PASTIS"
    s2_root = pastis_root / "S2_tiles_France"
    labels_root = pastis_root / "Labels"
    metadata_path = pastis_root / "metadata.geojson"
    folds = parse_pastis_folds(metadata_path)

    rows_by_key: dict[tuple, ManifestRow] = {}
    lookups: list[LookupRow] = []
    count = 0

    for tile_dir in iter_selected_tiles(s2_root, selected_tiles):
        tile = tile_dir.name
        label_dir = labels_root / tile
        labels = sorted(label_dir.glob("TARGET_*.tif"))
        s2_files = sorted(tile_dir.glob("*.tif"))
        if not labels or not s2_files:
            continue

        with rasterio.open(s2_files[0]) as ref:
            ref_transform = ref.transform

        label_infos = []
        for label_path in labels:
            patch_id = label_path.stem.split("_", 1)[1]
            fold = folds.get(str(int(patch_id)) if patch_id.isdigit() else patch_id)
            data_split = split_from_fold(fold, val_folds)
            with rasterio.open(label_path) as label_src:
                row_min, col_min = rasterio.transform.rowcol(
                    ref_transform,
                    label_src.bounds.left,
                    label_src.bounds.top,
                )
            label_infos.append((label_path, fold, data_split, int(col_min), int(row_min)))

        for s2_path in s2_files:
            timestamp = s2_path.stem
            embedding_year = year_from_timestamp(timestamp)
            with rasterio.open(s2_path) as s2_src:
                crs = s2_src.crs.to_string()
                for label_path, fold, data_split, x, y in label_infos:
                    height = 128
                    width = 128
                    window = Window(x, y, width, height)
                    transform = s2_src.window_transform(window)
                    grid_id = f"x{x}_y{y}_h{height}_w{width}"
                    sample_id = f"PASTIS_{tile}_{embedding_year}_{grid_id}"
                    row = ManifestRow(
                        sample_id=sample_id,
                        data_split=data_split,
                        embedding_path="",
                        label_path=str(label_path),
                        crs=crs,
                        geotransform=geotransform_from_transform(transform),
                        raster_shape=(height, width),
                        time_start=embedding_year,
                        lon=None,
                        lat=None,
                        tile=tile,
                        dataset="PASTIS",
                        image_timestamp=timestamp,
                        embedding_year=embedding_year,
                        source_s2_path=str(s2_path),
                        x=x,
                        y=y,
                        height=height,
                        width=width,
                        grid_id=grid_id,
                        fold=fold,
                    )
                    merged = merge_row(rows_by_key, row)
                    lookups.append(
                        LookupRow(
                            dataset="PASTIS",
                            tile=tile,
                            data_split=data_split,
                            image_timestamp=timestamp,
                            embedding_year=embedding_year,
                            sample_id=merged.sample_id,
                            grid_id=grid_id,
                            source_s2_path=str(s2_path),
                            label_path=str(label_path),
                            x=x,
                            y=y,
                            height=height,
                            width=width,
                            fold=fold,
                        )
                    )
                    count += 1
                    if limit and count >= limit:
                        return list(rows_by_key.values()), lookups

    return list(rows_by_key.values()), lookups


def load_burnscars_splits(csv_path: Path) -> dict[tuple[str, str], str]:
    splits = {}
    if not csv_path.exists():
        return splits
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            tile = row.get("tilename", "")
            timestamp = row.get("image_capture_date", "")
            raw_split = row.get("train_val", "")
            if raw_split == "training":
                split = "train"
            elif raw_split in {"validation", "val"}:
                split = "val"
            else:
                split = raw_split or "unknown"
            if tile and timestamp:
                splits[(tile, timestamp)] = split
    return splits


def has_positive_label(src, window) -> bool:
    arr = src.read(1, window=window, masked=False)
    return bool((arr > 0).any())


def build_hls_burnscars(
    data_root: Path,
    selected_tiles: set[str] | None,
    limit: int | None,
) -> tuple[list[ManifestRow], list[LookupRow]]:
    import rasterio
    from rasterio.windows import Window, bounds, from_bounds

    ds_root = data_root / "HLS_BrunScars"
    labels_root = ds_root / "Labels"
    s2_root = ds_root / "S2_tiles_USA"
    splits = load_burnscars_splits(ds_root / "tiles_metadata_with_10m_masks.csv")

    rows_by_key: dict[tuple, ManifestRow] = {}
    lookups: list[LookupRow] = []
    count = 0

    for tile_dir in iter_selected_tiles(labels_root, selected_tiles):
        tile = tile_dir.name
        for label_path in sorted(tile_dir.glob("*_mask_10m.tif")):
            timestamp = label_path.name.split("_mask_10m.tif")[0]
            s2_path = s2_root / tile / f"{timestamp}.tif"
            if not s2_path.exists():
                continue
            embedding_year = year_from_timestamp(timestamp)
            data_split = splits.get((tile, timestamp), "unknown")
            with rasterio.open(label_path) as label_src, rasterio.open(s2_path) as s2_src:
                crs = s2_src.crs.to_string()
                for y_label in range(0, label_src.height, 128):
                    for x_label in range(0, label_src.width, 128):
                        if x_label + 128 > label_src.width or y_label + 128 > label_src.height:
                            continue
                        label_window = Window(x_label, y_label, 128, 128)
                        if not has_positive_label(label_src, label_window):
                            continue

                        left, bottom, right, top = bounds(label_window, label_src.transform)
                        s2_window = from_bounds(left, bottom, right, top, transform=s2_src.transform)
                        s2_window = s2_window.round_offsets().round_lengths()
                        x = int(s2_window.col_off)
                        y = int(s2_window.row_off)
                        height = int(s2_window.height)
                        width = int(s2_window.width)
                        if height != 128 or width != 128:
                            continue

                        transform = s2_src.window_transform(s2_window)
                        grid_id = f"x{x}_y{y}_h{height}_w{width}"
                        sample_id = f"HLS_BrunScars_{tile}_{embedding_year}_{grid_id}"
                        row = ManifestRow(
                            sample_id=sample_id,
                            data_split=data_split,
                            embedding_path="",
                            label_path=str(label_path),
                            crs=crs,
                            geotransform=geotransform_from_transform(transform),
                            raster_shape=(height, width),
                            time_start=embedding_year,
                            lon=None,
                            lat=None,
                            tile=tile,
                            dataset="HLS_BrunScars",
                            image_timestamp=timestamp,
                            embedding_year=embedding_year,
                            source_s2_path=str(s2_path),
                            x=x,
                            y=y,
                            height=height,
                            width=width,
                            grid_id=grid_id,
                            fold=None,
                        )
                        merged = merge_row(rows_by_key, row)
                        lookups.append(
                            LookupRow(
                                dataset="HLS_BrunScars",
                                tile=tile,
                                data_split=data_split,
                                image_timestamp=timestamp,
                                embedding_year=embedding_year,
                                sample_id=merged.sample_id,
                                grid_id=grid_id,
                                source_s2_path=str(s2_path),
                                label_path=str(label_path),
                                x=x,
                                y=y,
                                height=height,
                                width=width,
                                fold=None,
                            )
                        )
                        count += 1
                        if limit and count >= limit:
                            return list(rows_by_key.values()), lookups

    return list(rows_by_key.values()), lookups


def find_bfp_label(data_root: Path, region: str) -> Path:
    matches = sorted((data_root / "BFP_Binary").glob(f"{region}_*_microsoft_buildings_2p5m.tif"))
    if not matches:
        raise FileNotFoundError(f"No BFP label found for region {region}")
    return matches[0]


def build_bfp(
    data_root: Path,
    selected_tiles: set[str] | None,
    limit: int | None,
) -> tuple[list[ManifestRow], list[LookupRow]]:
    import rasterio
    from rasterio.windows import Window

    bfp_root = data_root / "BFP_Binary"
    s2_root = data_root / "PASTIS" / "S2_tiles_France"
    sample_files = sorted(bfp_root.glob("*_samples_10perc.json"))

    rows_by_key: dict[tuple, ManifestRow] = {}
    lookups: list[LookupRow] = []
    count = 0

    for sample_file in sample_files:
        parts = sample_file.name.split("_")
        region = parts[0]
        if selected_tiles and region not in selected_tiles:
            continue
        data_split = "train" if "_train_" in sample_file.name else "val" if "_val_" in sample_file.name else "unknown"
        label_path = find_bfp_label(data_root, region)
        samples = json.loads(sample_file.read_text())
        for sample in samples:
            timestamp = Path(sample["time_str"]).stem
            embedding_year = year_from_timestamp(timestamp)
            x = int(sample["x"])
            y = int(sample["y"])
            height = 160
            width = 160
            s2_path = s2_root / region / sample["time_str"]
            if not s2_path.exists():
                continue
            with rasterio.open(s2_path) as s2_src:
                window = Window(x, y, width, height)
                transform = s2_src.window_transform(window)
                crs = s2_src.crs.to_string()

            grid_id = f"x{x}_y{y}_h{height}_w{width}"
            sample_id = f"BFP_{region}_{embedding_year}_{grid_id}"
            row = ManifestRow(
                sample_id=sample_id,
                data_split=data_split,
                embedding_path="",
                label_path=str(label_path),
                crs=crs,
                geotransform=geotransform_from_transform(transform),
                raster_shape=(height, width),
                time_start=embedding_year,
                lon=None,
                lat=None,
                tile=region,
                dataset="BFP",
                image_timestamp=timestamp,
                embedding_year=embedding_year,
                source_s2_path=str(s2_path),
                x=x,
                y=y,
                height=height,
                width=width,
                grid_id=grid_id,
                fold=None,
            )
            merged = merge_row(rows_by_key, row)
            lookups.append(
                LookupRow(
                    dataset="BFP",
                    tile=region,
                    data_split=data_split,
                    image_timestamp=timestamp,
                    embedding_year=embedding_year,
                    sample_id=merged.sample_id,
                    grid_id=grid_id,
                    source_s2_path=str(s2_path),
                    label_path=str(label_path),
                    x=x,
                    y=y,
                    height=height,
                    width=width,
                    fold=None,
                )
            )
            count += 1
            if limit and count >= limit:
                return list(rows_by_key.values()), lookups

    return list(rows_by_key.values()), lookups


def write_parquet(path: Path, records: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="snappy", use_dictionary=True)


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def copy_manifest_for_downloaders(base_path: Path) -> tuple[Path, Path]:
    stem = base_path.stem
    if stem.endswith("_base"):
        prefix = stem[:-5]
    else:
        prefix = stem
    tessera_path = base_path.with_name(f"{prefix}_tessera.parquet")
    alphaearth_path = base_path.with_name(f"{prefix}_alphaearth.parquet")
    shutil.copyfile(base_path, tessera_path)
    shutil.copyfile(base_path, alphaearth_path)
    return tessera_path, alphaearth_path


def group_by_output(rows: list[ManifestRow]) -> dict[str, list[ManifestRow]]:
    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[f"{row.dataset}_{row.tile}"].append(row)
    return dict(grouped)


def parse_csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {v.strip() for v in value.split(",") if v.strip()}


def parse_int_set(value: str) -> set[int]:
    return {int(v.strip()) for v in value.split(",") if v.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/dccstor/geofm-datasets/datasets/lianet_bench")
    parser.add_argument("--out-root", default="/dccstor/geofm-datasets/datasets/lianet_bench/embeddings/outputs/manifests")
    parser.add_argument("--dataset", default="all", choices=["all", "PASTIS", "HLS_BrunScars", "BFP"])
    parser.add_argument("--tiles", default=None, help="Comma-separated tile/region filter, e.g. T31TFM,T32ULU")
    parser.add_argument(
        "--grid-mode",
        default="windows",
        choices=["windows"],
        help="Only train/val window mode is supported in this implementation.",
    )
    parser.add_argument("--pastis-val-folds", default="1", help="Comma-separated PASTIS val folds for split labels")
    parser.add_argument("--limit", type=int, default=None, help="Limit raw lookup samples per dataset for smoke tests")
    parser.add_argument("--no-copy", action="store_true", help="Do not create Tessera/AlphaEarth manifest copies")
    args = parser.parse_args()

    require_deps()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    selected_tiles = parse_csv_set(args.tiles)
    val_folds = parse_int_set(args.pastis_val_folds)

    all_rows: list[ManifestRow] = []
    all_lookups: list[LookupRow] = []

    datasets = ["PASTIS", "HLS_BrunScars", "BFP"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        if dataset == "PASTIS":
            rows, lookups = build_pastis(data_root, val_folds, selected_tiles, args.limit)
        elif dataset == "HLS_BrunScars":
            rows, lookups = build_hls_burnscars(data_root, selected_tiles, args.limit)
        elif dataset == "BFP":
            rows, lookups = build_bfp(data_root, selected_tiles, args.limit)
        else:
            raise AssertionError(dataset)
        all_rows.extend(rows)
        all_lookups.extend(lookups)

    summary = []
    for name, rows in group_by_output(all_rows).items():
        records = [asdict(r) for r in sorted(rows, key=lambda r: (r.embedding_year, r.grid_id))]
        base_path = out_root / f"{name}_base.parquet"
        write_parquet(base_path, records)
        if not args.no_copy:
            copy_manifest_for_downloaders(base_path)
        summary.append(
            {
                "manifest": str(base_path),
                "dataset_tile": name,
                "rows": len(records),
                "years": ",".join(str(y) for y in sorted({r["embedding_year"] for r in records})),
            }
        )

    lookup_records = [asdict(r) for r in all_lookups]
    write_parquet(out_root / "lookup.parquet", lookup_records)
    write_csv(out_root / "lookup.csv", lookup_records)
    write_csv(out_root / "manifest_summary.csv", summary)

    print(f"Wrote {len(group_by_output(all_rows))} base manifests to {out_root}")
    print(f"Unique download rows: {len(all_rows)}")
    print(f"Lookup rows: {len(all_lookups)}")


if __name__ == "__main__":
    main()
