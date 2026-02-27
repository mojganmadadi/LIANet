import torch
import torch.nn as nn

import os
import json
from tqdm import tqdm

from torchmetrics import MetricTracker, MetricCollection
from metrics import multiclass_segmentation_metrics
from datasets import PASTIS, BurnScars
from models.terratorch_models.load_ckpt import load_terratorch_model
from helpers import compute_mean_std
import argparse
parser = argparse.ArgumentParser(
    description="Evaluate a single task with TerraTorch model."
)

parser.add_argument(
    "--task",
    default="PASTIS_T31TFM",
    choices=["PASTIS_T32ULU", "PASTIS_T31TFM", "BurnScars"]
)

parser.add_argument(
    "--ckpt_path",
    default="/home/user/results_local/finetuning_results/PASTIS_T31TFM/unet_valFolds3/2026-02-25_17-45-33/last.pt"
)

parser.add_argument(
    "--yaml_path",
    default="/home/user/results_local/finetuning_results/PASTIS_T31TFM/unet_valFolds3/2026-02-25_17-45-33/config.json"
)

parser.add_argument(
    "--method",
    default="unet"
)
args = parser.parse_args()


TASK = args.task
METHOD = args.method
ckpt_path = args.ckpt_path
yaml_path = args.yaml_path

TOP_DIR = "/home/user/data_shared"
S2_TILES = {"PASTIS_T32ULU": "T32ULU",
            "PASTIS_T31TFM": "T31TFM",
            "BurnScars": "T11SMT"}

TILE_SIZES = 10980

CUDA = True
BATCH_SIZE = 32
NUM_WORKERS = 4
USE_STANDARDIZATION = True

SAFE_TO_FILE = True
OUTPUTDIR = "/dccstor/geofm-pre/isawittmann/liayn_eval/results"

LABELS = {
          "PASTIS_T32ULU": "masks/PASTIS",
          "PASTIS_T31TFM": "masks/PASTIS",
          "BurnScars": "masks/BurnScars"}

NUM_CLASSES = {
    "PASTIS_T32ULU": 20,
    "PASTIS_T31TFM": 20,
    "BurnScars": 2
    }

# mean_std = compute_mean_std(os.path.join(TOP_DIR, S2_TILES[TASK]))
# print(f"Computed mean: {mean_std[0]}")
# print(f"Computed std: {mean_std[1]}")
STANDARDIZATION_MEANS = {
    "PASTIS_T32ULU": torch.tensor([
       1369.30456077, 1453.79652062, 1665.14520604, 1637.35233162,
       2080.30560776, 3240.00959187, 3685.71675205, 3846.82618007,
       3935.76541153, 4017.9926236 , 2969.84053526, 2230.21368633]),
    "PASTIS_T31TFM": torch.tensor([
       1336.76415598, 1434.2558045 , 1693.55212753, 1664.20803938,
       2165.69249867, 3410.45882799, 3856.87198206, 4025.54738893,
       4128.59612901, 4194.40404841, 3183.09358943, 2334.84550267]),
    "BurnScars": torch.tensor([
       1723.23727744, 1854.54370545, 2058.1006732 , 2196.62395424,
       2464.5957232 , 2914.3408868 , 3102.96172131, 3215.03134227,
       3253.63172794, 3269.92354348, 3277.31291962, 2800.31283209]),}

STANDARDIZATION_STDS = {
    "PASTIS_T32ULU": torch.tensor([
        504.46876456,  546.00943421,  557.46990616,  650.29818586,
        636.58028582,  804.69240557,  983.39656744, 1057.23404959,
       1008.34909228, 1127.41464981,  834.70442581,  765.65660432]),
    "PASTIS_T31TFM": torch.tensor([
        419.56071849,  469.03832626,  493.0953154 ,  599.90596214,
        578.90160698,  800.75412651,  972.14638009, 1043.85554465,
        999.6283347 , 1030.91240965,  802.18845332,  715.30376643
    ]),
    "BurnScars": torch.tensor([
        1053.96914392, 1029.80540471,  977.72731355, 1005.07139191,
        955.72662665,  928.9454876 ,  958.78638783, 1027.42694973,
        970.16355054,  956.37464955, 1006.44499828,  924.03973471
    ]),}
print(f"Loading model for task: {TASK}")
model = load_terratorch_model(ckpt_path=ckpt_path, yaml_path=yaml_path)
MODELDICT = {TASK: model}

# Run Evaluation for each tile size 
def run_evaluation_for_tile_size(tile_size):

    print(f"\n=== Evaluating with COMPLETE_TILESIZE={tile_size} ===")

    # Initialize dataset
    if TASK == "PASTIS_T32ULU" or TASK == "PASTIS_T31TFM":
        dataset = PASTIS(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES[TASK],
            labels=labels[TASK],
            train_val_key="val",
            val_folds=args.val_folds,
        )
    elif TASK == "BurnScars":
        dataset = BurnScars(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES[TASK],
            labels=labels[TASK],
            train_val_key="val",
        )
    else:
        raise ValueError(f"Unknown task: {TASK}")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, drop_last=False
    )

    # setup metrics
    list_of_metrics, maximize_list = multiclass_segmentation_metrics(
        num_classes=NUM_CLASSES[TASK],
        ignore_index=255 if TASK in ["PASTIS_T32ULU", "PASTIS_T31TFM"] else None
    )

    metrictracker = MetricCollection(list_of_metrics).to(device)
    model = MODELDICT[TASK].to(device)
    model.eval()

    # iterate over dataset
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {TASK} ({tile_size})"):
            inputs = batch["s2data"].to(device)
            labels = batch["label"].to(device)

            if USE_STANDARDIZATION:

                # raw_inputs = inputs * 10000.0 + 1000
                means = STANDARDIZATION_MEANS.view(1, -1, 1, 1).to(device)
                stds = STANDARDIZATION_STDS.view(1, -1, 1, 1).to(device)
                inputs = (inputs - means) / stds

            outputs_raw = model(inputs)
            outputs = getattr(outputs_raw, "output", outputs_raw)
            if outputs.dim() == 3:
                outputs = outputs.unsqueeze(1)
            metrictracker.update(outputs, labels)

    results = metrictracker.compute()
    print(f"\nResults for task={TASK}, tile_size={tile_size}")
    for name, value in sorted(results.items()):
        print(f"{name}: {float(value):.4f}")

    # Save results
    if SAFE_TO_FILE:
        save_dir = os.path.join(OUTPUTDIR, TASK, METHOD)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"metrics_{TASK}_{tile_size}.json")
        results_json = {n: float(v) for n, v in results.items()}
        with open(save_path, "w") as f:
            json.dump(results_json, f, indent=4)
        print(f"Saved metrics to: {save_path}")

    metrictracker.reset()

if __name__ == "__main__":
    device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")
    for tsize in TILE_SIZES:
        run_evaluation_for_tile_size(tsize)