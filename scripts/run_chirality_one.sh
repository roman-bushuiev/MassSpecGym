#!/bin/bash

# Single-baseline runner for the v1.5 chirality-baseline sweep.
# Reads env vars: CHIR_RUN_NAME, CHIR_MODEL, CHIR_CANDS,
#                 CHIR_FEAT (optional), CHIR_DIRECTION (optional, 'desc'|'asc').
# Submitted in parallel via submit_chirality_sweep.sh.

echo "job_key \"${job_key}\""
echo "run: ${CHIR_RUN_NAME} | model: ${CHIR_MODEL} | cands: ${CHIR_CANDS} | feat: ${CHIR_FEAT:-n/a} | dir: ${CHIR_DIRECTION:-n/a}"

cd /scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym
module use /appl/local/csc/modulefiles/
module load pytorch
source .venv/bin/activate

cd scripts

# Do not override CUDA_VISIBLE_DEVICES — SLURM has already mapped the
# allocated GPU into the job environment.

PKL_PATH="../data/test_results_v1.5/retrieval/${CHIR_RUN_NAME#v1.5_}.pkl"

EXTRA_ARGS=""
if [ -n "${CHIR_FEAT}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --chir_feature=${CHIR_FEAT}"
fi
if [ -n "${CHIR_DIRECTION}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --chir_direction=${CHIR_DIRECTION}"
fi

python3 run.py \
    --job_key="${job_key}" \
    --run_name="${CHIR_RUN_NAME}" \
    --devices=1 \
    --test_only \
    --task=retrieval \
    --model="${CHIR_MODEL}" \
    --dataset_pth="../data/MassSpecGym_RDKit_SMILES.tsv" \
    --candidates_pth="${CHIR_CANDS}" \
    --df_test_pth="${PKL_PATH}" \
    ${EXTRA_ARGS}
