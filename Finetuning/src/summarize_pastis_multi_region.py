#!/usr/bin/env python3
"""Summarize completed multi-region PASTIS fine-tuning runs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = ("jaccard_macro", "precision_macro", "f1_macro")
REGIONS = ("T31TFJ", "T32ULU", "T31TFM", "T30UXV")


def read_runs(root: Path):
    for metrics_path in root.rglob("best_metrics.json"):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists():
            continue
        with config_path.open() as stream:
            config = json.load(stream)
        task = config.get("task", "")
        if not task.startswith("PASTIS_joint_"):
            continue
        train_regions = config.get("train_regions")
        if not train_regions:
            train_regions = [
                region for region in REGIONS
                if region != task.removeprefix("PASTIS_joint_")
            ]
        region_metrics_path = metrics_path.parent / "best_metrics_by_region.json"
        if region_metrics_path.exists():
            with region_metrics_path.open() as stream:
                metrics_by_region = json.load(stream)
        else:
            with metrics_path.open() as stream:
                metrics_by_region = {
                    task.removeprefix("PASTIS_joint_"): json.load(stream).get(
                        "best_values", {}
                    )
                }
        for test_region, metrics in metrics_by_region.items():
            values = {
                metric: float(metrics[metric])
                for metric in METRICS
                if metric in metrics
            }
            if len(values) != len(METRICS):
                continue
            yield {
                "split": f"{len(train_regions)}to1",
                "test_region": test_region,
                "train_regions": ",".join(train_regions),
                "seed": int(config["seed"]),
                **values,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/user/results_local/finetuning_results/multi_region"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs = list(read_runs(args.root))
    if not runs:
        raise SystemExit(f"No completed runs with metrics found below {args.root}")

    expected = {"3to1": 20, "2to1": 60}
    counts = {}
    for run in runs:
        counts[run["split"]] = counts.get(run["split"], 0) + 1
    for split, expected_count in expected.items():
        print(f"{split}: found {counts.get(split, 0)}/{expected_count} completed runs")

    rows = []
    for split in ("3to1", "2to1"):
        split_runs = [run for run in runs if run["split"] == split]
        if not split_runs:
            continue
        for test_region in REGIONS:
            region_runs = [
                run for run in split_runs if run["test_region"] == test_region
            ]
            if not region_runs:
                continue
            row = {"split": split, "test_region": test_region, "runs": len(region_runs)}
            for metric in METRICS:
                row[metric] = mean(run[metric] for run in region_runs)
            rows.append(row)
        row = {"split": split, "test_region": "AVERAGE_ALL_TEST_REGIONS", "runs": len(split_runs)}
        for metric in METRICS:
            values = [run[metric] for run in split_runs]
            row[metric] = mean(values)
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        rows.append(row)

    print("\nBest validation metrics (macro averages; averaged over seeds and test regions):")
    print("split   test_region                 runs   IoU       Precision F1")
    for row in rows:
        print(
            f"{row['split']:5}   {row['test_region']:<27} {row['runs']:4}   "
            f"{row['jaccard_macro']:.4f}    {row['precision_macro']:.4f}    "
            f"{row['f1_macro']:.4f}"
        )

    output = args.output or args.root / "summary.csv"
    with output.open("w", newline="") as stream:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
