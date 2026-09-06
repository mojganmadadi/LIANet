#!/usr/bin/env bash
# Run the 20 three-source-region experiments (4 held-out regions x 5 seeds).

set -u

SEEDS=(42 111 222 333 444)
REGIONS=(T31TFJ T32ULU T31TFM T30UXV)
GPU_COUNT=8
BASE_DIR="${LOG_DIR:-/home/user/results_local/finetuning_results/3to1}"

run_experiment() {
    local gpu="$1" heldout="$2" seed="$3" train_regions regions_dir run_dir
    train_regions=()
    for region in "${REGIONS[@]}"; do
        [[ "$region" != "$heldout" ]] && train_regions+=("$region")
    done
    local train_list
    train_list=$(IFS=,; echo "${train_regions[*]}")
    regions_dir="${train_list//,/_}"
    run_dir="${BASE_DIR}/${heldout}/train_${regions_dir}/seed_${seed}"
    mkdir -p "$run_dir"

    python train.py \
        --config-name=PASTIS_LIANet \
        "task=PASTIS_joint_${heldout}" \
        "checkpoint_area=PASTIS_joint_${heldout}" \
        "train_regions=[${train_list}]" \
        "eval_regions=[${heldout}]" \
        val_folds=null \
        "seed=${seed}" \
        "gpu_id=${gpu}" \
        epochs=60 \
        validate_every_n_epochs=2 \
        plot_every_n_epochs=5 \
        batchsize=16 \
        num_workers=16 \
        "logging_directory=${run_dir}" \
        >"${run_dir}/run.log" 2>&1
}

jobs=()
for heldout in "${REGIONS[@]}"; do
    train_regions=()
    for region in "${REGIONS[@]}"; do
        [[ "$region" != "$heldout" ]] && train_regions+=("$region")
    done
    for seed in "${SEEDS[@]}"; do
        jobs+=("${heldout}|${seed}")
    done
done

worker() {
    local gpu="$1" index heldout seed
    for ((index=gpu; index<${#jobs[@]}; index+=GPU_COUNT)); do
        IFS='|' read -r heldout seed <<< "${jobs[$index]}"
        run_experiment "$gpu" "$heldout" "$seed"
    done
}

for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
    worker "$gpu" &
done
wait
echo "Finished 3-to-1 experiments."
