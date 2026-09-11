#!/bin/bash --login
#
# Build embedding manifests and pull AlphaEarth / Tessera annual embeddings.
#
# Usage:
#   bsub < download_lianet_embeddings_lsf.sh                    # all stages, all datasets
#   STAGES="manifests" bsub < download_lianet_embeddings_lsf.sh # manifests only
#   STAGES="alphaearth" STEMS="PASTIS_T31TFJ" bsub < ...        # one source, one tile
#
# Environment variables:
#   STAGES   space-separated subset of: manifests alphaearth tessera validate
#            (default: all four)
#   STEMS    space-separated manifest stems to pull
#            (default: all PASTIS + HLS_BrunScars + BFP tiles)
#   DATA_ROOT, EMBEDDING_ROOT, MANIFEST_DIR, EMBEDDING_DIR, QA_DIR, TESSERA_TMP_DIR
#   VENV_PATH, NWORKERS
#
#BSUB -n 4
#BSUB -M 20G
#BSUB -o lianet_pull_embeddings_%J.out
#BSUB -e lianet_pull_embeddings_%J.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
GET_EMBEDDINGS_DIR="${GET_EMBEDDINGS_DIR:-${PROJECT_ROOT}/get_embeddings}"
DATA_ROOT="${DATA_ROOT:-/dccstor/geofm-datasets/datasets/lianet_bench}"
EMBEDDING_ROOT="${EMBEDDING_ROOT:-${DATA_ROOT}/embeddings}"
MANIFEST_DIR="${MANIFEST_DIR:-${EMBEDDING_ROOT}/outputs/manifests}"
EMBEDDING_DIR="${EMBEDDING_DIR:-${EMBEDDING_ROOT}/outputs/embeddings}"
QA_DIR="${QA_DIR:-${EMBEDDING_ROOT}/outputs/qa}"
TESSERA_TMP_DIR="${TESSERA_TMP_DIR:-${EMBEDDING_ROOT}/tmp/tessera_tiles}"
NWORKERS="${NWORKERS:-4}"

STAGES="${STAGES:-manifests alphaearth tessera validate}"

# Tiles used by the cross-region benchmark. Override STEMS to pull a subset.
DEFAULT_STEMS="HLS_BrunScars_T11SMT HLS_BrunScars_T16REV BFP_T31TFM BFP_T32ULU \
PASTIS_T31TFM PASTIS_T32ULU PASTIS_T31TFJ PASTIS_T30UXV"
read -r -a manifest_stems <<< "${STEMS:-${DEFAULT_STEMS}}"

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

echo "Embedding download started: $(date)"
echo "Host          : $(hostname)"
echo "Project root  : ${PROJECT_ROOT}"
echo "Scripts       : ${GET_EMBEDDINGS_DIR}"
echo "Data root     : ${DATA_ROOT}"
echo "Embedding root: ${EMBEDDING_ROOT}"
echo "Stages        : ${STAGES}"
echo "Stems         : ${manifest_stems[*]}"

if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

export LIANET_DATA_DIR="${DATA_ROOT}"
export TMPDIR="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="${TMPDIR}/matplotlib"
mkdir -p "${MPLCONFIGDIR}" "${MANIFEST_DIR}" \
         "${EMBEDDING_DIR}/alphaearth" "${EMBEDDING_DIR}/tessera" \
         "${QA_DIR}" "${TESSERA_TMP_DIR}"

# ------------------------------------------------------------------ manifests
if has_stage manifests; then
  echo "=== Building manifests"
  python "${GET_EMBEDDINGS_DIR}/build_lianet_embedding_manifests.py" \
    --data-root "${DATA_ROOT}" \
    --out-root "${MANIFEST_DIR}"
fi

validate() {  # validate <manifest> <kind>
  has_stage validate || return 0
  echo "Validating $2: $1"
  python "${GET_EMBEDDINGS_DIR}/validate_embeddings.py" \
    --manifest "$1" \
    --kind "$2" \
    --embedding-dir "${EMBEDDING_ROOT}" \
    --out-dir "${QA_DIR}" \
    --sample-pixels 32
}

# ------------------------------------------------------------------ tessera
if has_stage tessera; then
  for stem in "${manifest_stems[@]}"; do
    manifest="${MANIFEST_DIR}/${stem}_tessera.parquet"
    echo "=== Pulling Tessera: ${manifest}"
    python "${GET_EMBEDDINGS_DIR}/download_tessera.py" \
      --manifest "${manifest}" \
      --out-dir "${EMBEDDING_DIR}/tessera" \
      --tmp-tiles-dir "${TESSERA_TMP_DIR}" \
      --nworkers "${NWORKERS}"
    validate "${manifest}" tessera
  done
fi

# ------------------------------------------------------------------ alphaearth
if has_stage alphaearth; then
  for stem in "${manifest_stems[@]}"; do
    manifest="${MANIFEST_DIR}/${stem}_alphaearth.parquet"
    echo "=== Pulling AlphaEarth: ${manifest}"
    python "${GET_EMBEDDINGS_DIR}/download_alphaearth.py" \
      --manifest "${manifest}" \
      --out-dir "${EMBEDDING_DIR}/alphaearth" \
      --nworkers "${NWORKERS}"
    validate "${manifest}" alphaearth
  done
fi

echo "Embedding download finished: $(date)"
