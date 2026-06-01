#!/bin/bash
#SBATCH --account=project_465002061
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --job-name=s4v2_bucket_emit
#SBATCH --output=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/bucket_emit_%j.out
#SBATCH --error=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/bucket_emit_%j.err

set -euo pipefail
module use /appl/local/csc/modulefiles/
module load pytorch
source /pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/.venv-genmol/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

WS=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev
SCRIPT=$WS/MassSpecGym/scripts/build_msg_s4_candidates.py
OUT_DIR=$WS/experiments/data_builds/MassSpecGym_S4_v2
DATA=$WS/MassSpecGym/data

# Emit the v2 S4-only candidate JSONs. These overwrite the stale ChEMBL-S4
# intermediates in data/ — the ChEMBL FINAL candidates are safely archived in
# data/v1.5_ChEMBL/. The augment + canon stages downstream read these paths.
COMMON="--out-dir $OUT_DIR --n-workers 128 \
        --out-formula-json $DATA/MassSpecGym_S4_retrieval_candidates_formula_nostereo.json \
        --out-mass-json    $DATA/MassSpecGym_S4_retrieval_candidates_mass_nostereo.json"

echo "=== STAGE: bucket ==="
python -u "$SCRIPT" --stage bucket $COMMON

echo "=== STAGE: emit_formula_json ==="
python -u "$SCRIPT" --stage emit_formula_json $COMMON

echo "=== STAGE: emit_mass_json ==="
python -u "$SCRIPT" --stage emit_mass_json $COMMON

echo "=== bucket + emit DONE ==="
