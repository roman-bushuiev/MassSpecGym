#!/bin/bash
# Sample 200M molecules from the S4 PRETRAIN-ONLY checkpoint (4M corpus, NO MSG-train finetune),
# matching the 59.6M-pool protocol (3 temps, dedup later) to isolate the finetune's effect on
# val-GT coverage. Loads MassSpecGym_S4_v2/ckpt_pretrain via --sample-ckpt.
#SBATCH --account=project_465003029
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=s4_pretrainonly_sample
#SBATCH --array=0-23
#SBATCH --output=/scratch/project_465003029/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_pretrainonly/sample_%A_%a.out

set -uo pipefail
WS=/scratch/project_465003029/rbushuie/DreaMS-Mol_dev
source "$WS/scripts/lib/load_env.sh" >/dev/null 2>&1
# caches off the home filesystem (home inode quota can be full -> EDQUOT / MIOpen segfault)
export XDG_CACHE_HOME=/scratch/project_465003029/rbushuie/.cache
export TORCH_HOME=$XDG_CACHE_HOME/torch MIOPEN_USER_DB_PATH=$XDG_CACHE_HOME/miopen
export MIOPEN_CUSTOM_CACHE_DIR=$XDG_CACHE_HOME/miopen TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
mkdir -p "$MIOPEN_USER_DB_PATH" "$TORCH_HOME" "$TRITON_CACHE_DIR"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

SCRIPT=$WS/MassSpecGym/scripts/v1.5_candidates/build_msg_s4_candidates.py
OUT_DIR=$WS/experiments/data_builds/MassSpecGym_S4_pretrainonly
SAMPLE_CKPT=$WS/experiments/data_builds/MassSpecGym_S4_v2/ckpt_pretrain
S4_REPO=$WS/s4-for-de-novo-drug-design

echo "=== S4 pretrain-only sample (200M, 3 temps, 24 shards) — task $SLURM_ARRAY_TASK_ID, host $(hostname) $(date) ==="
set +e
for attempt in 1 2 3 4; do
    echo "--- attempt $attempt $(date -Is) ---"
    python -u "$SCRIPT" --stage sample \
        --out-dir "$OUT_DIR" --sample-ckpt "$SAMPLE_CKPT" --s4-repo "$S4_REPO" \
        --n-designs 200000000 --temperatures "0.7,1.0,1.2" \
        --batch-size 2048 --chunk-size 200000 \
        --shard-id "$SLURM_ARRAY_TASK_ID" --n-shards 24
    rc=$?
    if [ "$rc" -eq 0 ]; then break; fi
    echo "[retry] attempt $attempt exited rc=$rc"; sleep 30
done
echo "DONE_RC=$? task=$SLURM_ARRAY_TASK_ID $(date)"
