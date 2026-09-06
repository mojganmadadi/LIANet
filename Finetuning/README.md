# Fine-tuning

The fine-tuning code supports the original single-region PASTIS experiments
and the region-aware multi-region experiments. The original `PASTIS_*` tasks
remain separate from `PASTIS_joint_*` tasks.

## Docker

Run commands from `/home/user/src` inside the fine-tuning container. The
container must mount this worktree's `Finetuning/src` directory at
`/home/user/src`, together with the data and results mounts described in
`docker/start_container.sh`.

Checkpoint paths and data paths are configured in `src/settings.py`. The
four-region pretrained checkpoint must contain `used_parameters.json` and
`model_checkpoints/latest_validation_checkpoint.pt`.

## Single-region PASTIS

The existing single-region command remains available:

```bash
python train_PASTIS.py
```

The main `train.py` entry point can also run a single-region task:

```bash
python train.py --config-name=PASTIS_LIANet \
  task=PASTIS_T31TFM \
  checkpoint_area=full_tile_modified_PASTIS_T31TFM
```

## Multi-region PASTIS

The fixed pretrained region IDs are:

```text
T31TFJ -> 0
T32ULU -> 1
T31TFM -> 2
T30UXV -> 3
```

To train on three regions and evaluate on `T32ULU`:

```bash
python train.py --config-name=PASTIS_LIANet \
  task=PASTIS_joint_T32ULU \
  checkpoint_area=PASTIS_joint_T32ULU \
  train_regions='[T31TFJ,T31TFM,T30UXV]' \
  eval_regions='[T32ULU]' \
  val_folds=null seed=42 gpu_id=0 epochs=60 \
  validate_every_n_epochs=2 plot_every_n_epochs=5 \
  batchsize=16 num_workers=16
```

For 2-to-1 experiments, provide two training regions and both unseen
evaluation regions:

```bash
train_regions='[T31TFJ,T32ULU]' \
eval_regions='[T31TFM,T30UXV]'
```

Validation is performed separately for each evaluation region. The reported
joint validation metrics are the equal arithmetic mean of the region metrics.
The run writes per-region metrics to `best_metrics_by_region.json` and
`best_steps_by_region.json`.

## Experiment launchers

The simple launchers use eight GPUs, one active training process per GPU:

```bash
nohup bash run_pastis_3_to_1.sh > run_3to1.log 2>&1 &
nohup bash run_pastis_2_to_1.sh > run_2to1.log 2>&1 &
```

Run them sequentially unless separate GPU allocations are configured.

After runs finish, summarize the per-region results:

```bash
python summarize_pastis_multi_region.py \
  --root /home/user/results_local/finetuning_results/3to1
python summarize_pastis_multi_region.py \
  --root /home/user/results_local/finetuning_results/2to1
```

Each run also stores the resolved Hydra configuration and `run_metadata.json`,
including the Git commit and fixed region-ID mapping.
