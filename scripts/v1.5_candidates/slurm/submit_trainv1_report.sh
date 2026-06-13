#!/bin/bash
#SBATCH --account=project_465003029
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --job-name=trainv1_cands_report
#SBATCH --output=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates/report_%j.out
#SBATCH --error=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates/report_%j.out

# Compute candidate statistics and render the HTML report. Run after Job C.
set -euo pipefail
source /pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev/scripts/lib/load_env.sh

RD=/pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev/experiments/reports/data-eda/2026-06-12_dreams-mol-train-v1-candidates

echo "===== compute stats (skip if pkl already present) ====="
if [ ! -f /scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates/candidate_stats.pkl ]; then
  python -u "$RD/scripts/compute_candidate_stats.py" --n-workers 56
else
  echo "  candidate_stats.pkl exists — skipping compute"
fi

echo "===== build report ====="
python -u "$RD/scripts/build_report.py"

echo "===== REPORT DONE ====="
ls -la "$RD"/2026-06-12_dreams-mol-train-v1-candidates.html "$RD"/figures/ 2>/dev/null | head