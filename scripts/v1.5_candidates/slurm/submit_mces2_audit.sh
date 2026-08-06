#!/bin/bash
# Audit the MCES2-disjointness level of the published MassSpecGym 4M pool (vs test) and of our
# val-filtered derivative (vs val). Karolina / SLURM, CPU-only.
#   sbatch --export=ALL,STAGE=quick submit_mces2_audit.sh   # probes A-D, ~40 min
#   sbatch --export=ALL,STAGE=mirror submit_mces2_audit.sh  # full builder rule applied to test
#SBATCH --account=open-37-54
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=/scratch/project/open-37-54/romanb/mces2_audit/%x_%j.out
set -euo pipefail

D=/scratch/project/open-37-54/romanb/mces2_audit
PY=/home/romanb/miniconda3/bin/python
POOL=$D/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv
MSG=$D/MassSpecGym1.5.tsv
REM=$D/removed_rows.tsv
W=128
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$D"

run() { echo; echo "################ $1 ################"; shift; "$PY" -u verify_mces2_disjointness.py "$@"; }

case "${STAGE:-quick}" in
quick)
  run "SELFTEST" --selftest --workers 8

  # A. Published pool vs TEST, identical-formula isomers only. MassSpecGym's stated rule is
  #    "exclude MCES < 2", so surviving pairs at exactly d == 2 are expected and pairs at
  #    d < 2 must be absent.
  run "A published-pool x TEST (delta=0)" \
      --pool "$POOL" --msg-tsv "$MSG" --fold test --prefilter withH --max-delta 0 \
      --workers $W --out $D/out/A_published_test_d0.json

  # B. Same probe against OUR val-filtered pool and the val fold. Our rule is "exclude
  #    MCES <= 2", so this must return zero hits.
  run "B ours x VAL (delta=0)" \
      --pool "$POOL" --drop-smiles "$REM" --msg-tsv "$MSG" --fold val \
      --prefilter withH --max-delta 0 --workers $W --out $D/out/B_ours_val_d0.json

  # C. Are the 8,304 removed molecules genuinely MCES <= 2 from val, and how many sit exactly
  #    on d == 2 (i.e. would have been KEPT under MassSpecGym's "< 2" rule)?
  run "C removed-set x VAL (delta=2)" \
      --pool "$REM" --msg-tsv "$MSG" --fold val --prefilter withH --max-delta 2 \
      --workers $W --out $D/out/C_removed_val_d2.json

  # D. Size the sound heavy-atom pre-filter before paying for it.
  run "D sizing: heavy prefilter, 100 val queries" \
      --pool "$POOL" --drop-smiles "$REM" --msg-tsv "$MSG" --fold val \
      --prefilter heavy --max-delta 2 --n-queries 100 --seed 0 --dry-run \
      --workers $W --out $D/out/D_sizing_heavy_val_100.json
  ;;
mirror)
  # The builder's exact procedure (with-H pre-filter, delta 2, MCES threshold 2) applied to the
  # TEST fold on the published pool: how many molecules would our val rule have removed that
  # MassSpecGym's test rule left in?
  run "MIRROR published-pool x TEST (delta=2, builder rule)" \
      --pool "$POOL" --msg-tsv "$MSG" --fold test --prefilter withH --max-delta 2 \
      --workers $W --out $D/out/MIRROR_published_test_d2.json
  ;;
hole)
  run "HOLE ours x VAL, heavy prefilter" \
      --pool "$POOL" --drop-smiles "$REM" --msg-tsv "$MSG" --fold val \
      --prefilter heavy --max-delta 2 --n-queries ${NQ:-100} --seed 0 \
      --workers $W --out $D/out/HOLE_ours_val_heavy_nq${NQ:-100}.json
  ;;
esac
echo "AUDIT_STAGE_DONE stage=${STAGE:-quick} rc=$?"
