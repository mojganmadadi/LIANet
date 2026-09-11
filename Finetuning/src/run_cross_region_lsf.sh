#!/bin/bash --login
#
# Submit one cross-region experiment to LSF.
#   CONFIG       required, path to a *_Auto.yaml automation config
#   PROJECT_ROOT repo root (default: this script's ../..)
#   VENV_PATH    virtualenv to activate (skipped if unset/absent)
#   DATA_ROOT    exported as LIANET_DATA_DIR    (dataset root)
#   RESULTS_ROOT exported as LIANET_RESULTS_DIR (overrides output_root in the config)
#   EXTRA_ARGS   extra OmegaConf dotlist overrides, e.g. "folds=[3] train.enabled=false"
#
#BSUB -n 4
#BSUB -M 20G
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o lianet_cross_region_%J.out
#BSUB -e lianet_cross_region_%J.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/dccstor/geofm-datasets/datasets/lianet_bench}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/Results/Finetuning_Automated}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${CONFIG:-}" ]]; then
  echo "ERROR: CONFIG is not set. Point it at a *_Auto.yaml automation config." >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: CONFIG does not exist: ${CONFIG}" >&2
  exit 2
fi

echo "Job started : $(date)"
echo "Host        : $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
echo "Config      : ${CONFIG}"
echo "Data root   : ${DATA_ROOT}"
echo "Results root: ${RESULTS_ROOT}"

if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
  echo "Virtualenv  : ${VENV_PATH}"
fi

export LIANET_DATA_DIR="${DATA_ROOT}"
export LIANET_RESULTS_DIR="${RESULTS_ROOT}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

cd "${PROJECT_ROOT}"

# shellcheck disable=SC2086
python Finetuning/src/run_cross_region_experiment.py \
  --config "${CONFIG}" \
  ${EXTRA_ARGS}

echo "Job finished: $(date)"
