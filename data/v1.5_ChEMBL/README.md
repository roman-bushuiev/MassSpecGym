# MassSpecGym v1.5 — final release files

Stereo-stripped, 2D-InChIKey-unique S4 retrieval candidates aligned with
canonicalised spectrum data, plus recomputed values for every column
that depends on `smiles`.

## Files

| File | Description |
|---|---|
| `MassSpecGym1.5.tsv` | 231,104 spectra × 14 metadata columns. `smiles` column is RDKit canonical + stereo-stripped (`MolToSmiles(canonical=True, isomericSmiles=False)`). Every row's `smiles` is a key in both candidate JSONs (100% coverage). **TSV is the canonical source of truth** — see `MassSpecGym1.5.mgf` below. |
| `MassSpecGym1.5.mgf` | Same 231,104 spectra in MGF format. **Derived from `MassSpecGym1.5.tsv` in a single pass** via `matchms.exporting.save_as_mgf` inside the same canon script (`MassSpecGym/scripts/fixes/rdkit_canon_massspecgym.py`). The two files therefore carry identical content. |
| `MassSpecGym1.5_retrieval_candidates_formula.json` | Per-query formula-filtered retrieval candidates. 28,936 keys. Each value is a list of up to 512 candidate SMILES; the query SMILES is at index 0; all candidates share the query's molecular formula and have **distinct 2D-InChIKeys** from each other and from the query. Built by S4 (200M generations, MSG-train-only fine-tune) + PubChem v1.5 fallback + Molpher RerouteBond morphs (for remaining short formula queries). |
| `MassSpecGym1.5_retrieval_candidates_mass.json` | Per-query mass-filtered (±10 ppm) retrieval candidates. 28,936 keys. Same structure; built by S4 + PubChem v1.5 fallback (no Molpher needed). |

## Pipeline (canon → mine → finalise)

The published v1.5 build pipeline runs in three phases:

1. **Canonicalise + recompute** (TSV → MGF):
   `MassSpecGym/scripts/fixes/rdkit_canon_massspecgym.py` reads
   `MassSpecGym/data/MassSpecGym.tsv`, writes
   `data/v1.5/MassSpecGym1.5.{tsv,mgf}`.
2. **Mine retrieval candidates** (TSV → JSONs):
   `MassSpecGym/scripts/build_msg_s4_pipeline.sh` chains the SLURM
   submit scripts under
   `experiments/data_builds/MassSpecGym_S4/` (emit → PubChem augment →
   Molpher augment), reading queries from
   `data/v1.5/MassSpecGym1.5.tsv` and writing
   `data/v1.5/MassSpecGym1.5_retrieval_candidates_{formula,mass}.json`.
3. **Canonicalise candidate values**:
   `MassSpecGym/scripts/fixes/canonicalise_v15_candidate_values.py`
   passes every candidate value through the same canonical+nostereo
   transformation as the TSV / JSON keys, eliminating the small
   fraction of Molpher-augmentation outputs with explicit-H notation.
   Idempotent.

## Standardisation + recomputation

`scripts/fixes/rdkit_canon_massspecgym.py` recomputes every column that
depends on `smiles`:

| column | recomputed from new SMILES |
|---|---|
| `smiles` | `Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)` |
| `formula` | `rdMolDescriptors.CalcMolFormula(mol)` |
| `inchikey` | first 14 chars of `InchiToInchiKey(MolToInchi(mol))` (the 2D-InChIKey convention used in v1) |
| `parent_mass` | `rdMolDescriptors.CalcExactMolWt(mol)` |
| `precursor_formula` | Hill-notation formula of (n_parents × new formula ± adduct ion components), via the same matchms adduct-parsing logic that built the v1 column |

All other columns (`identifier`, `mzs`, `intensities`, `precursor_mz`,
`adduct`, `instrument_type`, `collision_energy`, `fold`,
`simulation_challenge`) are passed through unchanged.

## Coverage / completeness

| Property | Value |
|---|---|
| Spectra in TSV | 231,104 |
| Spectra in MGF | 231,104 |
| Unique stripped SMILES in TSV | 28,936 |
| Queries in `candidates_formula.json` | 28,936 |
| Queries in `candidates_mass.json` | 28,936 |
| TSV rows whose `smiles` is a JSON key | **231,104 / 231,104 (100 %)** |
| Mean candidates per formula bucket | 389.25 |
| Mean candidates per mass bucket | 470.90 |
| Buckets exceeding 512-cap | 0 |
| Buckets with duplicate strings | 0 |
| Query SMILES at index 0 of every bucket | ✓ |
| Candidate values canonical (`MolToSmiles canonical=True, isomericSmiles=False`) | 99.99964 % (73 of 20,701,305 unique candidate strings flip between two equivalent forms under RDKit's own re-canonicalisation — pathological aromaticity edge cases, `canon(canon(s)) != canon(s)`. All parse to the correct molecule.) |

## Notebooks

- `notebooks/massspecgym_in_the_wild/candidates_analysis_final.ipynb`
  — full pipeline / per-step contribution / quality / per-formula and
  per-mass bucket-size distributions / qualitative examples /
  comparison to PubChem v1.5 / sanity checks.
- `notebooks/massspecgym_in_the_wild/v1.5_consistency_check.ipynb`
  — direct row-by-row comparison of v1.5 TSV/MGF vs the original
  PubChem-standardised `MassSpecGym.tsv` / `MassSpecGym.mgf`.
  Confirms:
  - 231,104 / 231,104 rows in both formats, no orphans.
  - All TSV diffs are confined to the 5 SMILES-dependent columns the canon script recomputes. Headline numbers:
    - `smiles` differs in 225,269 rows (97.475 %)
    - `formula` 0 (0.000 %) — stereo-stripping preserves molecular formula
    - `inchikey` 272 (0.118 %) — small differences from re-deriving 2D-IK
    - `parent_mass` 46,540 (20.138 %) — most are RDKit float-precision noise (median |Δ| ≈ 5e-5 Da); ~58 rows show |Δ| > 1 Da where the original release stored an incorrect mass and the recomputation corrects it
    - `precursor_formula` 58 (0.025 %) — adduct re-derivation edge cases
  - MGF peak lists 100 % identical.
  - MGF non-recomputed headers (`COLLISION_ENERGY`, `PRECURSOR_MZ`) show string-level diffs but max numeric residual 0.0 (matchms float `repr` reformat only).
