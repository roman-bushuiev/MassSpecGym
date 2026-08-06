#!/bin/bash
# Build MassSpecGym_molecules_MCES2_disjoint_with_valtest_fold_4M.tsv on Karolina.
#
#   sbatch --export=ALL,STAGE=prep     submit_mces_lt2_build.sh
#   sbatch --array=0-63 --export=ALL,STAGE=score,NSHARDS=64 submit_mces_lt2_build.sh
#   sbatch --export=ALL,STAGE=assemble,NSHARDS=64 submit_mces_lt2_build.sh
#
# `score` is resumable: each shard appends completed query indices to done_<shard>.txt and skips
# them on a re-run, so a walltime kill costs only the in-flight query.
#SBATCH --account=open-37-54
#SBATCH --partition=qcpu_free
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=17:00:00
#SBATCH --job-name=mceslt2
#SBATCH --output=/scratch/project/open-37-54/romanb/mces2_audit/%x_%A_%a.out
set -euo pipefail

D=/scratch/project/open-37-54/romanb/mces2_audit
PY=/home/romanb/miniconda3/bin/python
POOL=$D/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv
OUT=$D/build/MassSpecGym_molecules_MCES2_disjoint_with_valtest_fold_4M.tsv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$D"

case "${STAGE:-score}" in
prep)
  "$PY" -u build_mces_lt2_valtest_pool.py --stage prep \
      --pool "$POOL" --msg-tsv "$D/MassSpecGym1.5.tsv" \
      --cache "$D/build/heavy_cache.npz" --out-dir "$D/build" --workers 128
  ;;
score)
  "$PY" -u build_mces_lt2_valtest_pool.py --stage score \
      --pool "$POOL" --cache "$D/build/heavy_cache.npz" --out-dir "$D/build/shards" \
      --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${NSHARDS}" --workers 128
  ;;
assemble)
  "$PY" -u build_mces_lt2_valtest_pool.py --stage assemble \
      --pool "$POOL" --out-dir "$D/build/shards" --out "$OUT" --nshards "${NSHARDS}"
  ;;
esac
echo "MCESLT2_DONE stage=${STAGE:-score} rc=$?"
