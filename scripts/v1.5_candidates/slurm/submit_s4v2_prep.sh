#!/bin/bash
#SBATCH --account=project_465002061
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=s4v2_prep
#SBATCH --output=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/prep_%j.out
#SBATCH --error=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/experiments/data_builds/MassSpecGym_S4_v2/prep_%j.err

set -euo pipefail
module use /appl/local/csc/modulefiles/
module load pytorch
source /pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/.venv-genmol/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WS=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev
SCRIPT=$WS/MassSpecGym/scripts/fixes/prep_msg4m_pretrain_zips.py

echo "=== S4 v2: prep pretrain zips (MCES2-valtest-disjoint 4M + MSG-train) ==="
python -u "$SCRIPT" --workers 128
echo "=== DONE ==="
