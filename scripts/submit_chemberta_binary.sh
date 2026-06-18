#!/bin/bash

# Train + evaluate the ChemBERTa binary-classifier baseline on the formula
# challenge. Single SLURM allocation does training, inference, and MCES.

echo "job_key \"${job_key}\""

cd /scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym
module use /appl/local/csc/modulefiles/
module load pytorch
source .venv/bin/activate

cd scripts

# Do not override CUDA_VISIBLE_DEVICES — SLURM has already mapped the
# allocated GPU into the job environment.

EXTRA_ARGS=""
if [ -n "${SMOKE}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --smoke=${SMOKE}"
fi
if [ -n "${EPOCHS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --epochs=${EPOCHS}"
fi
if [ -n "${BATCH_SIZE}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --batch_size=${BATCH_SIZE}"
fi
if [ -n "${LOAD_CKPT}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --load_ckpt"
fi
if [ -n "${CHIRTSV_PTH}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --tsv_pth=${CHIRTSV_PTH}"
fi
if [ -n "${CHIRCANDS_PTH}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --cands_pth=${CHIRCANDS_PTH}"
fi
if [ -n "${CHIROUT_PKL}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --out_pkl=${CHIROUT_PKL}"
fi
if [ -n "${CHIRCKPT_DIR}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --ckpt_dir=${CHIRCKPT_DIR}"
fi

python3 -u train_eval_chemberta_binary.py ${EXTRA_ARGS}
