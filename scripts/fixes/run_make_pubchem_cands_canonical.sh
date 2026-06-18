#!/bin/bash
#SBATCH --job-name=canon_cands
#SBATCH --account=project_465002061
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

INPUT_JSON="${1:?Usage: sbatch run_make_pubchem_cands_canonical.sh <input_json>}"

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/.venv-genmol/bin/activate

cd /scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/scripts/fixes

python make_pubchem_cands_canonical.py "${INPUT_JSON}"
