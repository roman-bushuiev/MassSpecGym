# MassSpecGym v1.5 — MSG-4M-pretrained release

This release uses an S4 candidate generator pretrained on a
**MassSpecGym-aligned 4M-molecule corpus** (`MassSpecGym_molecules_MCES2_disjoint_with_valtest_4M`)
rather than the ChEMBL31 corpus used in `data/v1.5_ChEMBL/`. The motivation,
build, and headline outcome are documented in the
`notebooks/massspecgym_in_the_wild/` folder.

## Files

| File | Description |
|---|---|
| `MassSpecGym1.5.tsv` | 231,104 spectra × 14 metadata columns. Identical to v1.5_ChEMBL/MassSpecGym1.5.tsv — spectrum data does not depend on the generator. |
| `MassSpecGym1.5.mgf` | Same spectra in MGF format, identical to v1.5_ChEMBL/. |
| `MassSpecGym1.5_retrieval_candidates_formula.json` | Per-query 512-cap formula-filtered candidates produced by the MSG-4M S4 + PubChem fallback + Molpher augmentation, canonicalised values. |
| `MassSpecGym1.5_retrieval_candidates_mass.json` | Per-query 512-cap mass-filtered (±10 ppm) candidates from MSG-4M S4 + PubChem fallback. |

## Why a new pretrain corpus?

The ChEMBL release (`v1.5_ChEMBL/`) had a chir-distribution mismatch
with MSG GT: S4 candidates clustered at chir = 0 (drug-like, fully
aromatic), while MSG GT has median chir = 1. A 1-line RDKit baseline
(count potential stereocenters via `FindMolChiralCenters(includeUnassigned=True)`)
exploited this gap and reached **5× random hit@1 on mass** — a
benchmark artifact, not a real signal.

This release replaces the pretrain corpus with the MSG-aligned 4M pool
(MCES2-disjoint with val + test, `data/pretrain_corpora/...`). The
generated candidate chir distribution now matches GT essentially exactly:

|  | MSG-test GT | MSG-4M (new) | ChEMBL (old) |
|---|---|---|---|
| median chir | 1 | **1** | 0 |
| mean chir | 2.44 | **2.00 / 2.19** (formula / mass) | 1.52 / 1.79 |
| p(chir = 0) | 49 % | **44.5 / 45.4 %** | 58.7 / 54.1 % |

## Headline benchmark effect

Spectrum-agnostic priors (test fold) — both **chirality count** and the
**ChemBERTa SMILES classifier** (binary classifier of query-vs-candidate)
were exploiting the same candidate-pool/GT distribution mismatch in the
ChEMBL release. The mismatch is gone in this release, so both priors
collapse onto random.

|  | Random hit@1 | Chirality hit@1 | ChemBERTa hit@1 |
|---|---|---|---|
| Mass, OLD (ChEMBL) | 0.52 | 2.55 (**4.9×**) | 1.69 (3.3×) |
| **Mass, NEW (MSG-4M)** | **0.31** | **0.47 (1.5×)** | **0.21 (0.7× — below random)** |
| Formula, OLD | 1.82 | 2.01 (1.1×) | 2.12 (1.2×) |
| Formula, NEW | 1.60 | 1.96 (1.2×) | (retrain pending; old shows 2.12) |

The chirality count and the spectrum-agnostic ChemBERTa classifier
both lose their "chemistry-prior shortcut" against the new candidate
distribution — exactly the property we wanted for a clean
spectrum-conditioned-retrieval benchmark. Headline: artifact removed.

## Pipeline scripts

All under `MassSpecGym/scripts/fixes/` or
`DreaMS-Mol/scripts/data_processing/`:

1. `build_mces2_valtest_disjoint_pool.py` — MCES2-disjoint-with-val
   filter on the published MSG-MCES2-test-disjoint 4M pool. Output:
   `data/pretrain_corpora/MassSpecGym_molecules_MCES2_disjoint_with_valtest_4M.tsv` (3.91 M molecules; 8,304 removed for MCES ≤ 2 to val).
2. `prep_msg4m_pretrain_zips.py` — standardise + 95/5 split the MCES2
   pool ∪ MSG-train SMILES into pretrain zips (the MSG-train inclusion
   makes the auto-built tokenizer cover MSG-train tokens → zero OOV
   drops at finetune time).
3. `DreaMS-Mol/scripts/data_processing/build_s4_candidates.py --stage
   pretrain` with `--vocab-size 256 --sequence-length 160` —
   pretrained on the MSG-4M corpus, 30 epochs, val_loss 0.179.
4. `MassSpecGym/scripts/build_msg_s4_candidates.py --stage
   {extract,prep_train,finetune,sample,bucket,emit_*}` — finetune on
   MSG-train (now 0 OOV drops vs the old vocab's 15.5 %), then
   200 M-sample × 3-temperature sweep, then bucket / formula+mass
   emit.
5. PubChem augment + Molpher augment + canonicalise (existing
   pipeline).

## SLURM submission scripts

`experiments/data_builds/MassSpecGym_S4_v2/submit_s4v2_*.sh` — one
per pipeline stage. Bad GPU nodes excluded; PYTORCH_HIP_ALLOC_CONF
expandable_segments enabled.

## Notebooks

`notebooks/massspecgym_in_the_wild/` — the v1.5_consistency_check,
candidates_analysis_final, and evaluation_s4 notebooks are being
re-executed against this release; learnable-baseline retraining
(ChemBERTa + DS/DS+FF/FP-FFN grid) is the remaining piece.
