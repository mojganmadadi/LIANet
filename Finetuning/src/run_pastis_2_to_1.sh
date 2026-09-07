#!/usr/bin/env bash
# Run 30 two-source-region trainings (6 source pairs x 5 seeds).
# Each trained model is validated on both of the remaining regions.

set -u

SEEDS=(42 111 222 333 444)
REGIONS=(T31TFJ T32ULU T31TFM T30UXV)
GPU_COUNT=8
BASE_DIR="${LOG_DIR:-/home/user/results_local/finetuning_results/2to1}"

run_experiment() {
    local gpu="$1" train_a="$2" train_b="$3" test_a="$4" test_b="$5" seed="$6"
    local run_dir="${BASE_DIR}/train_${train_a}_${train_b}/test_${test_a}_${test_b}/seed_${seed}"
    mkdir -p "$run_dir"

    python train.py \
        --config-name=PASTIS_LIANet \
        "task=PASTIS_joint_${test_a}" \
        "checkpoint_area=PASTIS_joint_${test_a}" \
        "train_regions=[${train_a},${train_b}]" \
        "eval_regions=[${test_a},${test_b}]" \
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
for ((first=0; first<4; first++)); do
    for ((second=first+1; second<4; second++)); do
        test_regions=()
        for ((candidate=0; candidate<4; candidate++)); do
            if [[ "$candidate" != "$first" && "$candidate" != "$second" ]]; then
                test_regions+=("${REGIONS[$candidate]}")
            fi
        done
        for seed in "${SEEDS[@]}"; do
            jobs+=("${REGIONS[$first]}|${REGIONS[$second]}|${test_regions[0]}|${test_regions[1]}|${seed}")
        done
    done
done

worker() {
    local gpu="$1" index train_a train_b test_a test_b seed
    for ((index=gpu; index<${#jobs[@]}; index+=GPU_COUNT)); do
        IFS='|' read -r train_a train_b test_a test_b seed <<< "${jobs[$index]}"
        run_experiment "$gpu" "$train_a" "$train_b" "$test_a" "$test_b" "$seed"
    done
}

for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
    worker "$gpu" &
done
wait
echo "Finished 2-to-1 experiments: 6 trainings, each evaluated on 2 regions."
