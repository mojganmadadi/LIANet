"""Train and evaluate LIANet fine-tuning configs across source/target regions.
"""

import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchmetrics import MetricCollection
from tqdm import tqdm

from helpers import _embedding_dataset_and_tile, _embedding_manifest_for_task, load_model_class
from metrics import multiclass_segmentation_metrics, regression_metrics
from settings import TOP_DIR, activation_functions, labels, models, num_classes, s2_tiles


def _as_plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _format_override(key, value):
    value = _as_plain(value)
    if isinstance(value, list):
        return f"{key}=[{','.join(str(v) for v in value)}]"
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    return f"{key}={value}"


def _has_config_key(config, key):
    current = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _hydra_override_key(base_config, key):
    return key if _has_config_key(base_config, key) else f"+{key}"


def _task(config, tile):
    return config.task_template.format(tile=tile)


def _variants(config):
    base_overrides = _as_plain(config.train.get("extra_overrides", {})) or {}
    configured = config.train.get("variants")
    if configured:
        return [
            {
                "name": str(variant.get("name", f"variant_{index}")),
                "overrides": {
                    **base_overrides,
                    **(_as_plain(variant.get("overrides", {})) or {}),
                },
            }
            for index, variant in enumerate(configured)
        ]
    return [
        {
            "name": "default",
            "overrides": base_overrides,
        }
    ]


def _apply_variant(base_config, task, fold, seed, config, variant):
    base_config.update(variant["overrides"])
    base_config["task"] = task
    base_config["seed"] = int(seed)
    if config.train.get("pass_val_folds", True):
        base_config["val_folds"] = [int(fold)]
    return base_config


def _model_dir_name(train_config):
    model_type = train_config["model_type"]
    if model_type == "terratorch_factory":
        backbone = train_config["terratorch"]["model_args"]["backbone"]
        return f"terratorch_{backbone}_lr{train_config['learningrate']}_batchsize{train_config['batchsize']}"
    if model_type == "unet":
        if "PASTIS" in train_config["task"]:
            return (
                f"unet_valFolds{train_config['val_folds'][0]}_"
                f"lr{train_config['learningrate']}_batchsize{train_config['batchsize']}"
            )
        return f"unet_lr{train_config['learningrate']}_batchsize{train_config['batchsize']}"
    raise ValueError(f"Unsupported automated model_type: {model_type}")


def _run_root(config):
    # LIANET_RESULTS_DIR wins over the config value so the same config can be run on any
    # machine without editing it.
    output_root = os.environ.get("LIANET_RESULTS_DIR") or config.output_root
    return Path(output_root) / config.experiment_name


def _train_root(config):
    return _run_root(config) / "training_runs"


def _summary_root(config):
    return _run_root(config) / "summaries"


def _load_base_train_config(config):
    config_path = Path(__file__).resolve().parent / "configs"
    path = config_path / f"{config.base_train_config}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Base training config not found: {path}")
    with path.open("r") as f:
        return OmegaConf.to_container(OmegaConf.load(f), resolve=True)


def _expected_task_model_dir(config, task, fold, seed, variant):
    base = _load_base_train_config(config)
    base = _apply_variant(base, task, fold, seed, config, variant)
    return (
        _train_root(config)
        / f"variant_{variant['name']}"
        / f"fold_{fold}"
        / f"seed_{seed}"
        / task
        / _model_dir_name(base)
    )


def _latest_checkpoint(config, task, fold, seed, variant):
    model_dir = _expected_task_model_dir(config, task, fold, seed, variant)
    checkpoint_name = config.evaluation.checkpoint_name
    candidates = sorted(model_dir.glob(f"*/{checkpoint_name}"))
    if not candidates:
        return None
    return candidates[-1]


def _train_one(config, task, fold, seed, variant):
    existing = _latest_checkpoint(config, task, fold, seed, variant)
    if existing is not None and config.train.skip_existing:
        print(f"[train] skip existing {variant['name']} {task} fold={fold} seed={seed}: {existing}")
        return existing

    logging_directory = _train_root(config) / f"variant_{variant['name']}" / f"fold_{fold}" / f"seed_{seed}"
    logging_directory.mkdir(parents=True, exist_ok=True)

    command = [
        str(config.python_executable),
        "train.py",
        "--config-name",
        str(config.base_train_config),
        _format_override("task", task),
        _format_override("seed", int(seed)),
        _format_override("logging_directory", str(logging_directory)),
    ]
    if config.train.get("pass_val_folds", True):
        command.append(_format_override("val_folds", [int(fold)]))
    base_train_config = _load_base_train_config(config)
    for key, value in variant["overrides"].items():
        command.append(_format_override(_hydra_override_key(base_train_config, key), value))

    print(f"[train] launch {variant['name']} {task} fold={fold} seed={seed}")
    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)

    checkpoint = _latest_checkpoint(config, task, fold, seed, variant)
    if checkpoint is None:
        raise FileNotFoundError(
            f"Training finished but no {config.evaluation.checkpoint_name} found for "
            f"{task}, fold={fold}, seed={seed}."
        )
    return checkpoint


def _device(config):
    requested = str(config.evaluation.device)
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model_for_eval(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint.get("args")
    if train_config is None:
        config_path = Path(checkpoint_path).with_name("config.json")
        with config_path.open("r") as f:
            train_config = json.load(f)
    train_config = _as_plain(train_config)

    task = train_config["task"]
    model = load_model_class(
        task=task,
        model_type=train_config["model_type"],
        MODEL_PATH=models[task],
        NUM_CLASSES=num_classes[task],
        ACTIVATION_FUNCTION=activation_functions[task],
        TERRATORCH_CONFIG=train_config.get("terratorch"),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, train_config


def _regression_prediction_and_label(output, label):
    prediction = output.squeeze(1) if output.ndim == 4 and output.shape[1] == 1 else output
    target = label.squeeze(1) if label.ndim == 4 and label.shape[1] == 1 else label
    if prediction.shape != target.shape:
        raise ValueError(
            f"Regression prediction/target shape mismatch: "
            f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
        )
    if prediction.ndim > 2:
        prediction = prediction.reshape(-1)
        target = target.reshape(-1)
    return prediction, target


def _empty_class_counts():
    return {
        "target_class0_pixels": 0,
        "target_class1_pixels": 0,
        "pred_class0_pixels": 0,
        "pred_class1_pixels": 0,
        "ignored_pixels": 0,
    }


def _update_segmentation_diagnostics(diagnostics, output, target, ignore_index=None):
    if not torch.isfinite(output).all().item():
        diagnostics["nonfinite_output_batches"] += 1
    prediction = output.argmax(dim=1)
    valid = target != ignore_index if ignore_index is not None else torch.ones_like(target, dtype=torch.bool)
    diagnostics["ignored_pixels"] += int((~valid).sum().detach().cpu())
    for class_id in (0, 1):
        diagnostics[f"target_class{class_id}_pixels"] += int(((target == class_id) & valid).sum().detach().cpu())
        diagnostics[f"pred_class{class_id}_pixels"] += int(((prediction == class_id) & valid).sum().detach().cpu())


def _target_dataset(task, fold, train_config=None):
    input_source = train_config.get("input_source") if train_config else None
    if input_source in {"alphaearth", "tessera"}:
        from datasets import AnnualEmbeddingDataset

        dataset_name, tile = _embedding_dataset_and_tile(task)
        source_cfg = train_config[input_source]
        manifest_path = _embedding_manifest_for_task(
            source_cfg["manifest_path"],
            dataset_name,
            tile,
            input_source,
        )
        return AnnualEmbeddingDataset(
            lookup_path=source_cfg["lookup_path"],
            manifest_path=manifest_path,
            dataset=dataset_name,
            tile=tile,
            train_val_key="val",
            task=task,
            input_source=input_source,
            val_folds=[int(fold)] if "PASTIS" in task else None,
            cache_in_memory=source_cfg.get("cache_in_memory", False),
        )

    if "PASTIS" in task:
        from datasets import PASTIS

        return PASTIS(
            top_dir=TOP_DIR[task],
            s2_tiles=s2_tiles[task],
            labels=labels[task],
            train_val_key="val",
            val_folds=[int(fold)],
            compute_weights=False,
        )
    if "BurnScars" in task:
        from datasets import BurnScars

        return BurnScars(
            top_dir=TOP_DIR[task],
            s2_tiles=s2_tiles[task],
            labels=labels[task],
            train_val_key="val",
        )
    if "BFPDensity" in task:
        from datasets import BuildingCoverageRaster

        return BuildingCoverageRaster(
            top_dir=TOP_DIR[task],
            s2_tiles=s2_tiles[task],
            labels=labels[task],
            train_val_key="val",
        )
    if "BFPBinary" in task:
        from datasets import BuildingBinaryRaster

        return BuildingBinaryRaster(
            top_dir=TOP_DIR[task],
            s2_tiles=s2_tiles[task],
            labels=labels[task],
            train_val_key="val",
        )
    raise ValueError(f"Unsupported cross-region evaluation task: {task}")


def _evaluate_one(config, checkpoint_path, source_task, target_task, fold, seed, variant, device):
    model, train_config = _load_model_for_eval(checkpoint_path, device)
    task_type = "regression" if ("canopy_height" in target_task or "BFPDensity" in target_task) else "segmentation"

    if task_type == "regression":
        metrics_dict, _ = regression_metrics()
    else:
        metrics_dict, _ = multiclass_segmentation_metrics(
            num_classes=num_classes[target_task],
            ignore_index=255 if "PASTIS" in target_task else None,
        )
    metrics = MetricCollection(metrics_dict).to(device)
    diagnostics = _empty_class_counts() if task_type == "segmentation" else {}
    diagnostics["nonfinite_input_batches"] = 0
    diagnostics["nonfinite_output_batches"] = 0

    dataset = _target_dataset(target_task, fold, train_config=train_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.evaluation.batchsize),
        shuffle=False,
        num_workers=int(config.evaluation.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=True if int(config.evaluation.num_workers) > 0 else False,
    )

    with torch.no_grad():
        for batch in tqdm(
            loader,
            total=len(loader),
            desc=f"eval source={source_task} target={target_task} fold={fold} seed={seed}",
        ):
            x = batch["s2data"].to(device, non_blocking=device.type == "cuda")
            y = batch["label"].to(device, non_blocking=device.type == "cuda")
            if not torch.isfinite(x).all().item():
                diagnostics["nonfinite_input_batches"] += 1
            output = model(x)
            if task_type == "regression":
                prediction, target = _regression_prediction_and_label(output, y)
                if not torch.isfinite(output).all().item():
                    diagnostics["nonfinite_output_batches"] += 1
                metrics.update(prediction, target)
            else:
                target = y.long()
                _update_segmentation_diagnostics(
                    diagnostics,
                    output,
                    target,
                    ignore_index=255 if "PASTIS" in target_task else None,
                )
                metrics.update(output, target)

    values = {}
    for key, value in metrics.compute().items():
        values[key] = float(value.detach().cpu())

    row = {
        "source_task": source_task,
        "target_task": target_task,
        "source_tile": source_task.split("_")[-1],
        "target_tile": target_task.split("_")[-1],
        "fold": int(fold),
        "seed": int(seed),
        "variant_name": variant["name"],
        "learningrate": train_config.get("learningrate"),
        "batchsize": train_config.get("batchsize"),
        "checkpoint_path": str(checkpoint_path),
        "model_type": train_config["model_type"],
    }
    row.update(values)
    row.update(diagnostics)
    if task_type == "segmentation":
        print(
            "[eval diagnostics]",
            f"variant={variant['name']}",
            f"source={source_task}",
            f"target={target_task}",
            f"target_pixels=[{diagnostics['target_class0_pixels']},{diagnostics['target_class1_pixels']}]",
            f"pred_pixels=[{diagnostics['pred_class0_pixels']},{diagnostics['pred_class1_pixels']}]",
            f"ignored={diagnostics['ignored_pixels']}",
            f"nonfinite_input_batches={diagnostics['nonfinite_input_batches']}",
            f"nonfinite_output_batches={diagnostics['nonfinite_output_batches']}",
        )
    elif diagnostics["nonfinite_input_batches"] or diagnostics["nonfinite_output_batches"]:
        print(
            "[eval diagnostics]",
            f"variant={variant['name']}",
            f"source={source_task}",
            f"target={target_task}",
            f"nonfinite_input_batches={diagnostics['nonfinite_input_batches']}",
            f"nonfinite_output_batches={diagnostics['nonfinite_output_batches']}",
        )
    return row


def _smoke_test(config):
    tiles = list(config.tiles)
    if not tiles:
        raise ValueError("smoke test requires at least one tile")
    folds = list(config.folds)
    if not folds:
        raise ValueError("smoke test requires at least one fold")
    seeds = list(config.seeds)
    if not seeds:
        raise ValueError("smoke test requires at least one seed")

    source_task = _task(config, tiles[0])
    target_task = _task(config, tiles[-1])
    fold = int(folds[0])
    seed = int(seeds[0])
    variant = _variants(config)[0]

    base = _load_base_train_config(config)
    base = _apply_variant(base, source_task, fold, seed, config, variant)

    device = _device(config)
    print(f"[smoke] using device: {device}")
    print(f"[smoke] variant={variant['name']} source={source_task} target={target_task} fold={fold} seed={seed}")
    print(f"[smoke] expected model dir: {_expected_task_model_dir(config, source_task, fold, seed, variant)}")

    model = load_model_class(
        task=source_task,
        model_type=base["model_type"],
        MODEL_PATH=models[source_task],
        NUM_CLASSES=num_classes[source_task],
        ACTIVATION_FUNCTION=activation_functions[source_task],
        TERRATORCH_CONFIG=base.get("terratorch"),
    ).to(device)
    model.eval()

    dataset = _target_dataset(target_task, fold, train_config=base)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    metrics_dict, _ = multiclass_segmentation_metrics(
        num_classes=num_classes[target_task],
        ignore_index=255 if "PASTIS" in target_task else None,
    )
    metrics = MetricCollection(metrics_dict).to(device)

    max_batches = int(config.get("smoke_test", {}).get("max_batches", 1))
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            x = batch["s2data"].to(device)
            y = batch["label"].to(device).long()
            output = model(x)
            metrics.update(output, y)
            print(
                "[smoke] batch",
                batch_index,
                "input",
                tuple(x.shape),
                "label",
                tuple(y.shape),
                "output",
                tuple(output.shape),
            )

    values = {key: float(value.detach().cpu()) for key, value in metrics.compute().items()}
    row = {
        "source_task": source_task,
        "target_task": target_task,
        "source_tile": source_task.split("_")[-1],
        "target_tile": target_task.split("_")[-1],
        "fold": fold,
        "seed": seed,
        "variant_name": variant["name"],
        "learningrate": base.get("learningrate"),
        "batchsize": base.get("batchsize"),
        "checkpoint_path": "SMOKE_TEST_UNTRAINED_MODEL",
        "model_type": base["model_type"],
    }
    row.update(values)
    _summarize(config, [row])
    print(f"[smoke] metrics: {values}")
    print(f"[smoke] outputs written to {_summary_root(config)}")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clear_managed_summary_files(summary_dir):
    managed_names = [
        "detailed_metrics.csv",
        "average_by_source_target.csv",
        "average_by_source.csv",
        "average_by_target.csv",
        "average_by_variant_source_target.csv",
        "average_by_variant.csv",
        "global_average.csv",
        "cross_region_detailed_metrics.csv",
        "cross_region_detailed_by_variant.csv",
        "cross_region_average_by_variant.csv",
        "cross_region_average_by_source_target.csv",
        "cross_region_average_by_variant_source_target.csv",
        "cross_region_global_average.csv",
        "cross_region_best_variant.csv",
        "summary.json",
    ]
    for name in managed_names:
        path = summary_dir / name
        if path.is_file():
            path.unlink()


def _sort_rows(rows, keys):
    return sorted(rows, key=lambda row: tuple(row[key] for key in keys))


def _mean_rows(rows, group_keys, metric_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    out = []
    for group, group_rows in sorted(grouped.items()):
        result = {key: value for key, value in zip(group_keys, group)}
        result["n"] = len(group_rows)
        for metric in metric_keys:
            result[metric] = sum(float(row[metric]) for row in group_rows) / len(group_rows)
        out.append(result)
    return out


def _metric_higher_is_better(metric_name):
    lower_is_better_tokens = ("loss", "error", "mae", "mse", "rmse")
    return not any(token in metric_name.lower() for token in lower_is_better_tokens)


def _best_rows(rows, metric_name):
    if not rows:
        return []
    if metric_name not in rows[0]:
        available = sorted(key for key in rows[0] if key != "n")
        raise ValueError(
            f"Primary metric '{metric_name}' not found in summary rows. "
            f"Available metrics: {available}"
        )
    reverse = _metric_higher_is_better(metric_name)
    best = sorted(rows, key=lambda row: float(row[metric_name]), reverse=reverse)[0]
    best = dict(best)
    best["selection_metric"] = metric_name
    best["selection_mode"] = "max" if reverse else "min"
    return [best]


def _summarize(config, rows):
    summary_dir = _summary_root(config)
    summary_dir.mkdir(parents=True, exist_ok=True)
    _clear_managed_summary_files(summary_dir)

    if not rows:
        return

    metadata_keys = {
        "source_task",
        "target_task",
        "source_tile",
        "target_tile",
        "fold",
        "seed",
        "variant_name",
        "learningrate",
        "batchsize",
        "checkpoint_path",
        "model_type",
    }
    metric_keys = [key for key in rows[0].keys() if key not in metadata_keys]

    cross_rows = [row for row in rows if row["source_tile"] != row["target_tile"]]
    cross_rows = _sort_rows(cross_rows, ["variant_name", "source_tile", "target_tile", "fold", "seed"])
    if not cross_rows:
        raise ValueError(
            "No cross-region evaluation rows were produced. "
            "Use at least two tiles/tasks or enable cross-target evaluation."
        )

    cross_by_pair = _mean_rows(cross_rows, ["source_tile", "target_tile"], metric_keys)
    cross_by_variant_pair = _mean_rows(
        cross_rows,
        ["variant_name", "source_tile", "target_tile"],
        metric_keys,
    )
    cross_by_variant = _mean_rows(cross_rows, ["variant_name"], metric_keys)
    cross_global_rows = _mean_rows(cross_rows, ["model_type"], metric_keys)
    best_variant = _best_rows(cross_by_variant, str(config.summary.primary_metric))

    _write_csv(summary_dir / "cross_region_detailed_by_variant.csv", cross_rows)
    _write_csv(summary_dir / "cross_region_average_by_variant.csv", cross_by_variant)
    _write_csv(summary_dir / "cross_region_average_by_source_target.csv", cross_by_pair)
    _write_csv(summary_dir / "cross_region_average_by_variant_source_target.csv", cross_by_variant_pair)
    _write_csv(summary_dir / "cross_region_global_average.csv", cross_global_rows)
    _write_csv(summary_dir / "cross_region_best_variant.csv", best_variant)

    payload = {
        "experiment_name": str(config.experiment_name),
        "primary_metric": str(config.summary.primary_metric),
        "num_detailed_rows": len(rows),
        "num_cross_region_rows": len(cross_rows),
        "cross_region_global_average": cross_global_rows,
        "cross_region_best_variant": best_variant,
    }
    with (summary_dir / "summary.json").open("w") as f:
        json.dump(payload, f, indent=4)


def run(config):
    required_sections = ["train", "evaluation", "summary"]
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"Automation config missing required sections: {missing}")

    run_root = _run_root(config)
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "automation_config_resolved.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))

    tasks = [_task(config, tile) for tile in config.tiles]
    variants = _variants(config)
    checkpoints = {}

    if config.train.enabled:
        for variant in variants:
            for fold in config.folds:
                for seed in config.seeds:
                    for source_task in tasks:
                        checkpoints[(variant["name"], source_task, int(fold), int(seed))] = _train_one(
                            config, source_task, int(fold), int(seed), variant
                        )
    else:
        for variant in variants:
            for fold in config.folds:
                for seed in config.seeds:
                    for source_task in tasks:
                        checkpoint = _latest_checkpoint(config, source_task, int(fold), int(seed), variant)
                        if checkpoint is None:
                            raise FileNotFoundError(
                                f"Missing checkpoint for variant={variant['name']} "
                                f"{source_task}, fold={fold}, seed={seed}."
                            )
                        checkpoints[(variant["name"], source_task, int(fold), int(seed))] = checkpoint

    rows = []
    if config.evaluation.enabled:
        device = _device(config)
        print(f"[eval] using device: {device}")
        for variant in variants:
            for fold in config.folds:
                for seed in config.seeds:
                    for source_task in tasks:
                        checkpoint = checkpoints[(variant["name"], source_task, int(fold), int(seed))]
                        for target_task in tasks:
                            if (
                                not bool(config.evaluation.include_self_region)
                                and source_task == target_task
                            ):
                                continue
                            rows.append(
                                _evaluate_one(
                                    config,
                                    checkpoint,
                                    source_task,
                                    target_task,
                                    int(fold),
                                    int(seed),
                                    variant,
                                    device,
                                )
                            )

    _summarize(config, rows)
    print(f"[done] outputs written to {_summary_root(config)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "PASTIS_TerraMind_Embedding_CrossRegion_Auto.yaml"),
        help="Path to the automation YAML config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides, e.g. folds=[3] train.enabled=false.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a one-batch model/dataset/metric/summary smoke test without training.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    if args.smoke_test:
        _smoke_test(config)
    else:
        run(config)


if __name__ == "__main__":
    main()
