#!/bin/bash
#SBATCH --account=project_465003029
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --job-name=trainv1_cands_molpher
#SBATCH --output=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates/molpher_%j.out
#SBATCH --error=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates/molpher_%j.out

# Job B: fill formula queries still <8 after S4+PubChem with Molpher RerouteBond
# morphs (formula-preserving). Uses the molpher-lib conda env (legacy tree, by
# abspath). Resumable via checkpoint. Outputs S4plusPCplusMol formula JSON.
set -euo pipefail
SC=/pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev/MassSpecGym/scripts/v1.5_candidates
RW=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates
NW=56

source /scratch/project_465002044/rbushuie/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/molpher/.venv
python -c "import molpher; from molpher.core import MolpherMol; import rdkit; print('molpher env OK, rdkit', rdkit.__version__)"
python -c "import os; print('available cores:', len(os.sched_getaffinity(0)))"

echo "===== molpher fill (formula only) ====="
WORKERS=$NW python -u "$SC/augment_s4_with_molpher.py" \
    --in-json "$RW/s4pc_formula.json" \
    --out-json "$RW/s4pcmol_formula.json" \
    --checkpoint "$RW/s4pcmol_formula_checkpoint.json" \
    --workers $NW

echo "===== JOB B DONE — S4plusPCplusMol formula ready; next: finalize (Job C) ====="
