#!/bin/bash
#SBATCH --account=project_465003029
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --job-name=trainv1_cands_main
#SBATCH --output=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates/main_%j.out
#SBATCH --error=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates/main_%j.out

# Job A: extract -> S4 emit (formula+mass) -> PubChem (index+emit, restricted to
# S4-short queries) -> augment with PubChem (formula+mass). Reuses the existing
# 409.8M S4 pool (buckets.parquet) by symlink. Output: S4plusPC nostereo JSONs.
set -euo pipefail
source /pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev/scripts/lib/load_env.sh
python -c "import os; print('available cores:', len(os.sched_getaffinity(0)))"

ROOT=/pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev
SC=$ROOT/MassSpecGym/scripts/v1.5_candidates
TSV=$ROOT/DreaMS-Mol/data/DreaMS-Mol_train_v1.tsv
POOL=$ROOT/experiments/data_builds/MassSpecGym_S4_v3_1B/buckets.parquet
PUBCHEM=/scratch/project_465003029/data/MassSpecGym/pubchem_inchi.tsv
WORK=$ROOT/experiments/data_builds/DreaMS-Mol_train_v1_candidates
RW=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates
NW=56
mkdir -p "$WORK" "$RW"

echo "===== [1] extract unique query molecules ====="
python -u "$SC/build_msg_s4_candidates.py" --stage extract --input-tsv "$TSV" --out-dir "$WORK" --n-workers $NW --force

echo "===== [2] symlink the reused 409.8M S4 pool ====="
ln -sf "$POOL" "$WORK/buckets.parquet"
ls -lL "$WORK/buckets.parquet"

echo "===== [3] S4 emit formula + mass ====="
python -u "$SC/build_msg_s4_candidates.py" --stage emit_formula_json --out-dir "$WORK" \
    --out-formula-json "$RW/s4_formula.json" --n-workers $NW
python -u "$SC/build_msg_s4_candidates.py" --stage emit_mass_json --out-dir "$WORK" \
    --out-mass-json "$RW/s4_mass.json" --n-workers $NW

echo "===== [4] PubChem candidates (index + emit, restricted to S4-short) ====="
python -u "$SC/build_pubchem_candidates.py" --stage all \
    --queries-parquet "$WORK/extract_unique_mols.parquet" \
    --pubchem-tsv "$PUBCHEM" --index-parquet "$RW/pubchem_index.parquet" \
    --out-formula "$RW/pc_formula.json" --out-mass "$RW/pc_mass.json" \
    --restrict-formula-json "$RW/s4_formula.json" --restrict-mass-json "$RW/s4_mass.json" \
    --threshold 8 --n-workers $NW

echo "===== [5] augment S4 with PubChem (formula + mass) ====="
WORKERS=$NW python -u "$SC/augment_s4_with_pubchem.py" \
    --s4-formula "$RW/s4_formula.json" --s4-mass "$RW/s4_mass.json" \
    --pc-formula "$RW/pc_formula.json" --pc-mass "$RW/pc_mass.json" \
    --out-formula "$RW/s4pc_formula.json" --out-mass "$RW/s4pc_mass.json" \
    --n-workers $NW

echo "===== JOB A DONE — S4plusPC ready; next: molpher (Job B) ====="
