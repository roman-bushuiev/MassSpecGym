#!/bin/bash
#SBATCH --account=project_465002061
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=nid005306,nid005379,nid005931,nid005932,nid007166,nid007228,nid007382,nid007488,nid005174,nid005176,nid007343,nid007068,nid006353,nid006158,nid006155,nid006550,nid007364,nid005537,nid006352,nid007655
#SBATCH --job-name=s4v2_sample
#SBATCH --output=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/sample_%A_%a.out
#SBATCH --error=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/sample_%A_%a.err
#SBATCH --array=0-7

# 200M samples across T in {0.7, 1.0, 1.2}, 8 shards. Loads v2 ckpt_finetune_msg
# (vocab_size=256, sequence_length=160 via from_file). batch_size reduced to 2048
# (vs 4096) because seq_len=160 > the ChEMBL pretrain's 99 → higher per-sample mem.

set -euo pipefail
module use /appl/local/csc/modulefiles/
module load pytorch
source /pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/.venv-genmol/bin/activate
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

WS=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev
SCRIPT=$WS/MassSpecGym/scripts/build_msg_s4_candidates.py
OUT_DIR=$WS/experiments/data_builds/MassSpecGym_S4_v2
N_SHARDS=8
SHARD_ID=${SLURM_ARRAY_TASK_ID}

echo "=== S4 v2 sample (200M, 3 temps × 8 shards) — task $SHARD_ID / $N_SHARDS ==="
echo "Node: $(hostname)"

set +e
for attempt in 1 2 3 4; do
    echo "--- sample attempt $attempt $(date -Is) shard=$SHARD_ID ---"
    python -u "$SCRIPT" \
        --stage sample \
        --out-dir "$OUT_DIR" \
        --n-designs 200000000 \
        --temperatures "0.7,1.0,1.2" \
        --batch-size 2048 \
        --chunk-size 200000 \
        --shard-id "$SHARD_ID" \
        --n-shards "$N_SHARDS"
    rc=$?
    if [ "$rc" -eq 0 ]; then break; fi
    echo "[retry] attempt $attempt exited rc=$rc"
    sleep 30
done
set -e

echo "=== sample task $SHARD_ID DONE ==="
