#!/bin/bash
#SBATCH --job-name=rdkit_canon_msg
#SBATCH --account=project_465002061
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/.venv-genmol/bin/activate

cd /scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/scripts/fixes

python rdkit_canon_massspecgym.py
