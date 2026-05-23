#!/bin/bash
# End-to-end reproducible build pipeline for MassSpecGym v1.5.
#
# Pipeline order (published / canon-first):
#
#   0. canon           local, ~2m    — rdkit_canon_massspecgym.py:
#                                      read data/MassSpecGym.tsv, canonicalise
#                                      SMILES + recompute every SMILES-derived
#                                      column, write
#                                      data/v1.5/MassSpecGym1.5.{tsv,mgf}
#                                      (single source of truth for spectra).
#   1. emit            CPU 1.5h      — emit_formula + emit_mass (uses existing
#                                      buckets.parquet, no resampling).
#                                      Reads queries from the canon TSV
#                                      data/v1.5/MassSpecGym1.5.tsv.
#   2. pubchem_aug     CPU 1h        — PubChem v1.5 augmentation for short
#                                      formula / mass lists.
#   3. molpher_aug     CPU 4h        — Molpher RerouteBond morphs for
#                                      remaining short formula lists.
#   4. build_caches    CPU 30m       — Morgan-FP + 2D-InChIKey caches over
#                                      the canon TSV.
#
# Run from anywhere:
#   bash MassSpecGym/scripts/build_msg_s4_pipeline.sh
#
# The realign_tsv stage that existed in the older "mine-first" version of
# this pipeline is no longer needed — the canon TSV produced by stage 0 is
# already aligned with the JSON keys produced by stages 1-3.
#
# The S4 sampling stages (extract / prep_train / finetune / sample / bucket)
# are not re-run here — they produce the 64M-mol pool at
# experiments/data_builds/MassSpecGym_S4/buckets.parquet, which is
# bucket-shape-only and thus reusable across cap settings. Re-running them
# is multi-day GPU work documented in CONSTRUCTION.md.
set -euo pipefail

WS=/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev
EXP=$WS/experiments/data_builds/MassSpecGym_S4
SCRIPTS=$WS/MassSpecGym/scripts
DATA=$WS/MassSpecGym/data

echo "0) canon (local, blocking — must finish before any mining stage starts)"
(cd $WS/MassSpecGym && python -u scripts/fixes/rdkit_canon_massspecgym.py)
echo "   data/v1.5/MassSpecGym1.5.{tsv,mgf} written"

echo "1) emit (reads queries from canon TSV)"
JID_EMIT=$(sbatch --parsable $EXP/submit_msg_s4_emit_nostereo.sh)
echo "   $JID_EMIT"

echo "2) PubChem augment (after emit)"
JID_PC=$(sbatch --parsable --dependency=afterok:$JID_EMIT $EXP/submit_msg_s4_augment.sh)
echo "   $JID_PC"

echo "3) Molpher augment (after PubChem)"
JID_MOL=$(sbatch --parsable --dependency=afterok:$JID_PC $EXP/submit_msg_s4_molpher.sh)
echo "   $JID_MOL"

echo "4) Build mol caches (after Molpher)"
JID_CACHE=$(sbatch --parsable --dependency=afterok:$JID_MOL $EXP/submit_build_caches.sh)
echo "   $JID_CACHE"

cat <<EOF

submitted chain:
  canon         (local, done synchronously before sbatch submissions)
  emit          $JID_EMIT
  pubchem_aug   $JID_PC     depends_on $JID_EMIT
  molpher_aug   $JID_MOL    depends_on $JID_PC
  build_caches  $JID_CACHE  depends_on $JID_MOL

Monitor with: squeue -u \$USER

Outputs:
  $DATA/v1.5/MassSpecGym1.5.tsv
  $DATA/v1.5/MassSpecGym1.5.mgf
  $DATA/v1.5/MassSpecGym1.5_retrieval_candidates_formula.json
  $DATA/v1.5/MassSpecGym1.5_retrieval_candidates_mass.json
  $EXP/caches/mol_fp_4096.h5
  $EXP/caches/mol_ik2d.pkl
EOF
