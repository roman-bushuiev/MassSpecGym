#!/usr/bin/env bash
#SBATCH --job-name=v16_verify
#SBATCH --account=project_465003029
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail
source /pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev/scripts/lib/load_env.sh
PY="${DREAMSMOL_WORK}/.venv-genmol/bin/python"
SCRIPT="${DREAMSMOL_ROOT}/MassSpecGym/scripts/v1.6_candidates/verify_v16_disjoint.py"
MSG="${DREAMSMOL_DATA}/MassSpecGym"

echo "Start: $(date)  cores=${SLURM_CPUS_PER_TASK:-?}"
"${PY}" "${SCRIPT}" \
    --v16-root "${MSG}" --tsv "${MSG}/v1.5/MassSpecGym1.5.tsv" --v15-dir "${MSG}/v1.5" \
    --criteria inchi tani80 --tani-threshold 0.80 --n-workers "${SLURM_CPUS_PER_TASK:-32}"
echo "Done: $(date)"
