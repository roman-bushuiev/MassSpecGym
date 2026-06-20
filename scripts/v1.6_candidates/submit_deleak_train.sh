#!/usr/bin/env bash
#SBATCH --job-name=v16_deleak
#SBATCH --account=project_465003029
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

# Train-only deleak: keep v1.5 val/test verbatim, clean train decoys of val/test-decoy overlap
# (inchi=2D-InChIKey, tani80=Tanimoto>=0.80) and backfill train to cap 512 from the S4 pool.
# Overwrites data/MassSpecGym/v1.6{inchi,tani80}/ with the train-only-deleaked sets.

set -euo pipefail
source /pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev/scripts/lib/load_env.sh
PY="${DREAMSMOL_WORK}/.venv-genmol/bin/python"
SCRIPT="${DREAMSMOL_ROOT}/MassSpecGym/scripts/v1.6_candidates/deleak_train_candidates.py"
MSG="${DREAMSMOL_DATA}/MassSpecGym"
BUCKETS="${DREAMSMOL_ROOT}/experiments/data_builds/MassSpecGym_S4_v3_1B/buckets.parquet"

echo "Start: $(date)  cores=${SLURM_CPUS_PER_TASK:-?}"
"${PY}" "${SCRIPT}" \
    --v15-dir "${MSG}/v1.5" --tsv "${MSG}/v1.5/MassSpecGym1.5.tsv" --mgf "${MSG}/v1.5/MassSpecGym1.5.mgf" \
    --buckets "${BUCKETS}" --out-root "${MSG}" \
    --criteria inchi tani80 --tani-threshold 0.80 --n-workers "${SLURM_CPUS_PER_TASK:-32}"
echo "Done: $(date)"
echo "Outputs: ${MSG}/v1.6inchi/ and ${MSG}/v1.6tani80/ (val/test = v1.5 verbatim; train deleaked+backfilled)"
