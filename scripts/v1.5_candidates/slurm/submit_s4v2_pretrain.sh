#!/bin/bash
#SBATCH --account=project_465002061
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=nid005306,nid005379,nid005931,nid005932,nid007166,nid007228,nid007382,nid007488,nid005174,nid005176,nid007343,nid007068,nid006353,nid006158,nid006155,nid006550,nid007364,nid005537,nid006352,nid007655
#SBATCH --job-name=s4v2_pretrain
#SBATCH --output=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/pretrain_%j.out
#SBATCH --error=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/pretrain_%j.err

# Pretrain S4 on the MCES2-valtest-disjoint 4M MSG corpus (mixed with MSG-train SMILES
# for vocab coverage at finetune time). Uses the same DreaMS-Mol/scripts/data_processing/
# build_s4_candidates.py --stage pretrain that the ChEMBL pretrain used, but with the
# new corpus zips (still named chembl_std_{train,valid}.zip for pipeline compatibility)
# and a new out_dir so we don't clobber the existing ChEMBL ckpt.

set -euo pipefail
module use /appl/local/csc/modulefiles/
module load pytorch
source /pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/.venv-genmol/bin/activate

# Reduce HIP allocator fragmentation (recommended in the OOM error message).
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

WS=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev
SCRIPT=$WS/DreaMS-Mol/scripts/data_processing/build_s4_candidates.py
OUT_DIR=$WS/experiments/data_builds/MassSpecGym_S4_v2

echo "=== S4 v2 pretrain on MCES2-valtest-disjoint MSG-4M + MSG-train ==="
echo "Node: $(hostname)"
rocm-smi --showproductname 2>/dev/null || nvidia-smi -L 2>/dev/null || true

CKPT_DIR=$OUT_DIR/ckpt_pretrain

# Same retry-loop as the ChEMBL pretrain (build_s4_candidates writes per-epoch ckpts
# and resumes on retry; useful on GPU faults).
set +e
for attempt in 1 2 3 4 5 6 7 8; do
    echo "--- pretrain attempt $attempt $(date -Is) ---"
    python -u "$SCRIPT" \
        --stage pretrain \
        --out-dir "$OUT_DIR" \
        --epochs 30 \
        --batch-size 512 \
        --vocab-size 256 \
        --sequence-length 160
    rc=$?
    if [ -d "$CKPT_DIR" ] && [ -f "$CKPT_DIR/model.pt" ]; then
        echo "[done] $CKPT_DIR/model.pt present after attempt $attempt (rc=$rc)"
        break
    fi
    echo "[retry] attempt $attempt exited rc=$rc; resuming from latest epoch ckpt"
    sleep 10
done
set -e

echo "=== pretrain DONE ==="
