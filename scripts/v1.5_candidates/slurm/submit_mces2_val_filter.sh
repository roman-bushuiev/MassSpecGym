#!/bin/bash
#SBATCH --account=project_465002061
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --job-name=mces2_val_filter
#SBATCH --output=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/mces2_val_filter_%j.out
#SBATCH --error=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/mces2_val_filter_%j.err

set -euo pipefail
module use /appl/local/csc/modulefiles/
module load pytorch
source /pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/.venv/bin/activate

# Cap BLAS threads per worker so 128 workers stay under RLIMIT_NPROC.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WS=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev
SCRIPT=$WS/MassSpecGym/scripts/fixes/build_mces2_valtest_disjoint_pool.py
POOL=$WS/MassSpecGym/data/pretrain_corpora/MSG_MCES2_test_disjoint_4M.tsv
MSG_TSV=$WS/MassSpecGym/data/v1.5/MassSpecGym1.5.tsv
OUT=$WS/MassSpecGym/data/pretrain_corpora/MassSpecGym_molecules_MCES2_disjoint_with_valtest_4M.tsv

mkdir -p "$(dirname "$OUT")"

# Move the input pool to the project data dir if it's still in /tmp.
if [ ! -f "$POOL" ]; then
    if [ -f /tmp/MSG_MCES2_test_disjoint_4M.tsv ]; then
        echo "Copying /tmp pool to project data dir ..."
        cp /tmp/MSG_MCES2_test_disjoint_4M.tsv "$POOL"
    else
        echo "ERROR: pool not found at $POOL"
        exit 1
    fi
fi

echo "=== MCES2-val filter ==="
echo "  pool:    $POOL"
echo "  msg-tsv: $MSG_TSV"
echo "  out:     $OUT"
echo "  workers: 128 (cpus-per-task)"

python -u "$SCRIPT" \
    --pool "$POOL" \
    --msg-tsv "$MSG_TSV" \
    --out "$OUT" \
    --workers 128

echo "=== DONE ==="
