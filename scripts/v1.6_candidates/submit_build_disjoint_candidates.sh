#!/usr/bin/env bash
#SBATCH --job-name=msg_v16_cands
#SBATCH --account=project_465003029
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

# Build MassSpecGym v1.6 cross-fold-disjoint candidates (formula + mass) for BOTH criteria
# (inchi = exact 2D-InChIKey; tani80 = Tanimoto>=0.80) -> data/MassSpecGym/v1.6inchi/ and v1.6tani80/.
# CPU-only work on a LUMI-G node (CPU budget exhausted); .venv-genmol has rdkit + pandas + pyarrow.

set -euo pipefail
source /pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev/scripts/lib/load_env.sh

PY="${DREAMSMOL_WORK}/.venv-genmol/bin/python"
SCRIPT="${DREAMSMOL_ROOT}/MassSpecGym/scripts/v1.6_candidates/build_disjoint_candidates.py"
TSV="${DREAMSMOL_DATA}/MassSpecGym/v1.5/MassSpecGym1.5.tsv"
BUCKETS="${DREAMSMOL_ROOT}/experiments/data_builds/MassSpecGym_S4_v3_1B/buckets.parquet"
OUT_ROOT="${DREAMSMOL_DATA}/MassSpecGym"
NP="${SLURM_CPUS_PER_TASK:-32}"

echo "Start: $(date)  node=$(hostname)  cores=${NP}"
"${PY}" "${SCRIPT}" \
    --tsv "${TSV}" --buckets "${BUCKETS}" --out-root "${OUT_ROOT}" \
    --criteria inchi tani80 --tani-threshold 0.80 --n-workers "${NP}"
echo "Done: $(date)"
echo "Outputs:"
echo "  ${OUT_ROOT}/v1.6inchi/MassSpecGym1.6_retrieval_candidates_{formula,mass}.json"
echo "  ${OUT_ROOT}/v1.6tani80/MassSpecGym1.6_retrieval_candidates_{formula,mass}.json"
