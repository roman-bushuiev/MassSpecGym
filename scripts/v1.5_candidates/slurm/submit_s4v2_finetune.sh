#!/bin/bash
#SBATCH --account=project_465002061
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nid005306,nid005379,nid005931,nid005932,nid007166,nid007228,nid007382,nid007488,nid005174,nid005176,nid007343,nid007068,nid006353,nid006158,nid006155,nid006550,nid007364,nid005537,nid006352,nid007655
#SBATCH --job-name=s4v2_finetune
#SBATCH --output=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/finetune_%j.out
#SBATCH --error=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/finetune_%j.err

# Finetune the v2 (MCES2-valtest-disjoint MSG-4M) pretrain on MSG-train SMILES.
# Runs extract + prep_train + finetune in sequence on the v2 out-dir, loading the
# v2 ckpt_pretrain (vocab_size=256, sequence_length=160 restored via from_file).
# Because the v2 pretrain corpus already contained MSG-train, prep_train's OOV-drop
# vs the v2 pretrain vocab should be ~0 (vs 15.5% under the old ChEMBL vocab).

set -euo pipefail
module use /appl/local/csc/modulefiles/
module load pytorch
source /pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/.venv-genmol/bin/activate
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

WS=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev
SCRIPT=$WS/MassSpecGym/scripts/build_msg_s4_candidates.py
OUT_DIR=$WS/experiments/data_builds/MassSpecGym_S4_v2
PRETRAIN_CKPT=$OUT_DIR/ckpt_pretrain
INPUT_TSV=$WS/MassSpecGym/data/v1.5/MassSpecGym1.5.tsv
CKPT_FINAL=$OUT_DIR/ckpt_finetune_msg

COMMON="--out-dir $OUT_DIR --pretrain-ckpt $PRETRAIN_CKPT --input-tsv $INPUT_TSV --batch-size 512"

echo "=== S4 v2: extract ==="
python -u "$SCRIPT" --stage extract $COMMON

echo "=== S4 v2: prep_train (OOV-drop vs v2 vocab should be ~0) ==="
python -u "$SCRIPT" --stage prep_train $COMMON

echo "=== S4 v2: finetune on MSG-train ==="
set +e
for attempt in 1 2 3 4 5 6 7 8; do
    echo "--- finetune attempt $attempt $(date -Is) ---"
    python -u "$SCRIPT" --stage finetune --epochs 50 $COMMON
    rc=$?
    if [ -d "$CKPT_FINAL" ] && [ -f "$CKPT_FINAL/model.pt" ]; then
        echo "[done] ckpt_finetune_msg present after attempt $attempt (rc=$rc)"
        break
    fi
    echo "[retry] attempt $attempt exited rc=$rc; resuming from latest epoch ckpt"
    sleep 10
done
set -e

echo "=== finetune DONE ==="
