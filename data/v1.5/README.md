# MassSpecGym v1.5 — final release files

Stereo-stripped, 2D-InChIKey-unique S4 retrieval candidates aligned with
canonicalised spectrum data.

## Files

| File | Description |
|---|---|
| `MassSpecGym1.5.tsv` | 231,104 spectra × 14 metadata columns. `smiles` column is RDKit canonical + stereo-stripped (`MolToSmiles(canonical=True, isomericSmiles=False)`). Every row's `smiles` is a key in both candidate JSONs (100% coverage). **TSV is the canonical source of truth** — see `MassSpecGym1.5.mgf` below. |
| `MassSpecGym1.5.mgf` | Same 231,104 spectra in MGF format. **Derived from `MassSpecGym1.5.tsv` in a single pass** via `matchms.exporting.save_as_mgf` (script: `MassSpecGym/scripts/fixes/build_mgf_from_tsv.py`). The two files therefore carry identical content — TSV columns become MGF headers, `mzs`/`intensities` become the peak list. |
| `MassSpecGym1.5_retrieval_candidates_formula.json` | Per-query formula-filtered retrieval candidates. 28,936 keys. Each value is a list of up to 512 candidate SMILES; the query SMILES is at index 0; all candidates share the query's molecular formula and have **distinct 2D-InChIKeys** from each other and from the query. Built by S4 (200M generations, MSG-train-only fine-tune) + PubChem v1.5 fallback (for queries with <8 S4 candidates) + Molpher RerouteBond morphs (for remaining short formula queries). |
| `MassSpecGym1.5_retrieval_candidates_mass.json` | Per-query mass-filtered (±10 ppm) retrieval candidates. 28,936 keys. Same structure; built by S4 + PubChem v1.5 fallback (no Molpher needed — 0 % trivial after step 2). |

## Standardisation

All SMILES strings — TSV `smiles`, MGF `SMILES=` headers, JSON keys, and
JSON candidate values — share **one canonical form**:

```python
from rdkit import Chem
canonical_nostereo = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s),
                                                 canonical=True,
                                                 isomericSmiles=False)
```

This makes the join keys exact (no implicit canonicalisation needed by
the consumer). Verified by `notebooks/massspecgym_in_the_wild/v1.5_consistency_check.ipynb`.

## Coverage / completeness

| Property | Value |
|---|---|
| Spectra in TSV | 231,104 |
| Spectra in MGF | 231,104 |
| Unique stripped SMILES in TSV | 28,922 |
| Queries in `candidates_formula.json` | 28,936 |
| Queries in `candidates_mass.json` | 28,936 |
| TSV rows whose `smiles` is a JSON key | **231,104 / 231,104 (100 %)** |
| Formula final — trivial (len=1) | 0 (0.000 %) |
| Mass final — trivial (len=1) | 0 (0.000 %) |
| Per-bucket 2D-InChIKey uniqueness (100 sampled buckets) | **0 duplicates** |

(The 14 JSON-key "extras" — `28,936 − 28,922` — are alternative
canonical forms of 2D structures that already have a TSV-aligned key
in the JSON; they don't cause coverage problems because no spectrum
ever has those as its `smiles`. They were emitted by the canonical
form chosen by the build pipeline.)

## Notebooks

- `notebooks/massspecgym_in_the_wild/candidates_analysis_final.ipynb`
  — full pipeline / per-step contribution / quality / per-formula and
  per-mass bucket-size distributions / qualitative examples /
  comparison to PubChem v1.5 / sanity checks (incl. SMILES
  canonicalisation parity across TSV / MGF / JSONs).
- `notebooks/massspecgym_in_the_wild/v1.5_consistency_check.ipynb`
  — direct row-by-row comparison of v1.5 TSV/MGF vs the original
  PubChem-standardised `MassSpecGym.tsv` / `MassSpecGym.mgf`. Confirms:
  - 231,104 / 231,104 rows in both formats, no orphans.
  - **TSV**: `smiles` differs in 97.48% of rows (PubChem-canonical → RDKit-canonical + stereo-stripped, as designed); all other columns identical.
  - **MGF**: peak lists 100% identical; `SMILES` differs in the same 97.48% as the TSV; `FORMULA` / `INCHIKEY` identical; `COLLISION_ENERGY` / `PARENT_MASS` / `PRECURSOR_MZ` show sub-1e-6 numeric round-off differences (`matchms.exporting.save_as_mgf` reformats floats — e.g. `41.490019999999994` → `41.49002` — the numeric values are unchanged).

## Provenance

Build pipeline (re-runnable):
`MassSpecGym/scripts/build_msg_s4_pipeline.sh`

It chains the SLURM submit scripts under
`experiments/data_builds/MassSpecGym_S4/`:
emit → PubChem v1.5 augment → Molpher augment → TSV realign →
MGF realign. The pool (`buckets.parquet`) is the same 64,205,194-mol
S4 sample produced by `expmisc004` (200M raw / 3 temperatures /
1,008 SLURM-array shards).
