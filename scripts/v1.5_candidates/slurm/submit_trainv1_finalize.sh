#!/bin/bash
#SBATCH --account=project_465003029
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --job-name=trainv1_cands_final
#SBATCH --output=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates/finalize_%j.out
#SBATCH --error=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates/finalize_%j.out

# Job C: canonicalise candidate values (formula = S4+PC+Mol, mass = S4+PC),
# re-key from nostereo to exact train_v1 SMILES, verify, place final JSONs in
# the DreaMS-Mol candidates dir (symlinked into DreaMS-Mol/data/candidates).
set -euo pipefail
source /pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev/scripts/lib/load_env.sh

ROOT=/pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev
SC=$ROOT/MassSpecGym/scripts/v1.5_candidates
DM=$ROOT/DreaMS-Mol/scripts/data_processing
TSV=$ROOT/DreaMS-Mol/data/DreaMS-Mol_train_v1.tsv
RW=/scratch/project_465003029/data/reports_work/dreams-mol-train-v1-candidates
CAND=/scratch/project_465003029/data/DreaMS-Mol/candidates
NW=56
mkdir -p "$CAND"

echo "===== [1] ensure s4pcmol_formula.json, then stage final nostereo JSONs ====="
# Molpher (Job B) writes s4pcmol_formula.json on success. If it was killed
# (e.g. a pathological RerouteBond morph hung the C++ core), assemble the set
# from the resumable checkpoint: S4+PubChem updated with the per-query molpher
# fills completed so far. Queries past the last checkpoint stay at S4+PubChem.
if [ ! -f "$RW/s4pcmol_formula.json" ]; then
  echo "  s4pcmol_formula.json absent — assembling from s4pc_formula + molpher checkpoint"
  python - <<PY
import json
s4pc = json.load(open("$RW/s4pc_formula.json"))
try:
    ckpt = json.load(open("$RW/s4pcmol_formula_checkpoint.json"))
    s4pc.update(ckpt)
    print(f"  applied {len(ckpt):,} molpher-filled queries from checkpoint")
except FileNotFoundError:
    print("  no molpher checkpoint; using s4pc_formula as-is")
json.dump(s4pc, open("$RW/s4pcmol_formula.json", "w"))
PY
fi
cp "$RW/s4pcmol_formula.json" "$RW/final_formula_nostereo.json"
cp "$RW/s4pc_mass.json"       "$RW/final_mass_nostereo.json"

echo "===== [2] canonicalise candidate values (in place) ====="
python -u "$SC/canonicalise_v15_candidate_values.py" \
    --formula-json "$RW/final_formula_nostereo.json" \
    --mass-json    "$RW/final_mass_nostereo.json"

echo "===== [3] re-key nostereo -> exact train_v1 SMILES ====="
python -u "$DM/rekey_candidates_to_trainv1.py" --tsv "$TSV" \
    --in-formula "$RW/final_formula_nostereo.json" --in-mass "$RW/final_mass_nostereo.json" \
    --out-formula "$CAND/DreaMS-Mol_train_v1_retrieval_candidates_formula.json" \
    --out-mass    "$CAND/DreaMS-Mol_train_v1_retrieval_candidates_mass.json" \
    --n-workers $NW

echo "===== [4] verify ====="
python -u "$DM/verify_candidates.py" --tsv "$TSV" \
    --formula-json "$CAND/DreaMS-Mol_train_v1_retrieval_candidates_formula.json" \
    --mass-json    "$CAND/DreaMS-Mol_train_v1_retrieval_candidates_mass.json" \
    --sample 2000 --max-decoys 80

echo "===== JOB C DONE — final candidates in $CAND ====="
ls -la "$CAND"/DreaMS-Mol_train_v1_retrieval_candidates_*.json
