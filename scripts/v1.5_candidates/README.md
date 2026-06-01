# MassSpecGym v1.5 candidate-pool generation pipeline

This folder contains every script used to build the
`MassSpecGym1.5_retrieval_candidates_{formula,mass}.json` candidate pools
that ship in `data/v1.5/`. It is the **MSG-4M release** — the S4 generator
is pretrained on a MassSpecGym-aligned 4 M-molecule corpus
(MCES2-disjoint with val+test), not on ChEMBL.

A higher-level write-up of the motivation + the headline benchmark effect
(both chirality and ChemBERTa priors collapse on the new candidates) is in
[`data/v1.5/README.md`](../../data/v1.5/README.md).

---

## TL;DR for the collaborator

**Goal:** replace the current **uniform-random** trim of per-query candidate
buckets with a **similarity-weighted** sampler (some fraction of candidates
drawn by Tanimoto/MCS similarity to the query).

**The only files you need to touch:**

| File | Lines | What's there |
|---|---|---|
| `build_msg_s4_candidates.py` | **622–625** | `_trim_uniform(rng, candidates, cap)` — the trim function |
| `build_msg_s4_candidates.py` | **665, 707** | the two call sites: `_stage_emit_formula_json` and `_stage_emit_mass_json` |
| `augment_s4_with_pubchem.py` | search for `random.sample` | second-stage trim after PubChem augment (CAP=512) |
| `augment_s4_with_molpher.py` | search for `random.sample` | third-stage trim after Molpher augment (CAP=512) |

The query SMILES is already in scope at each call site (it's the key being
iterated over, e.g. `q_smi` / `q_formula` / `q_ik` in the loop).

**The input the sampler sees** is just a Python `list[str]` of candidate
SMILES — already deduped by 2D-InChIKey, already canonicalised, already
excluded the query itself. Replace `rng.sample(candidates, cap)` with
your similarity-weighted variant. Suggested signature:

```python
def _trim_similarity_weighted(
    rng, candidates: list[str], cap: int,
    query_smiles: str, weight_by: str = "tanimoto"
) -> list[str]:
    if len(candidates) <= cap: return candidates
    # 1. compute fingerprint(query) once
    # 2. for each c in candidates: w_c = sim(query, c) (Tanimoto on
    #    Morgan-1024 is a sensible default; cheap to vectorise via RDKit)
    # 3. sample `cap` candidates from `candidates` with probability ∝ w_c
    #    (or top-K, or a mix — design decision)
    return chosen
```

**To test your sampler end-to-end** you only need to re-run the **`emit_*`
stages** of `build_msg_s4_candidates.py`:

```bash
python build_msg_s4_candidates.py --stage emit_formula_json \
    --out-dir <local_dir_with_buckets.parquet_and_extract_unique_mols.parquet> \
    --out-formula-json MassSpecGym_S4_retrieval_candidates_formula_nostereo.json \
    --n-workers 64

python build_msg_s4_candidates.py --stage emit_mass_json \
    --out-dir <same_dir> \
    --out-mass-json MassSpecGym_S4_retrieval_candidates_mass_nostereo.json \
    --n-workers 64
```

Then re-run `augment_s4_with_pubchem.py` + `augment_s4_with_molpher.py`
+ `canonicalise_v15_candidate_values.py` to regenerate the final
`MassSpecGym1.5_retrieval_candidates_{formula,mass}.json`.

The pretrain/finetune/sample stages **do not need to be re-run** — the
deduped pool (`buckets.parquet`, 59.6 M rows) is the only input the
sampler consumes.

**Data files you'll be sent separately** (~2 GB total):

| File | Purpose |
|---|---|
| `buckets.parquet` | the deduped S4 pool (59.6 M SMILES with `inchikey_2d, formula, exact_mass, log_likelihood, temperature`). The candidate universe. |
| `extract_unique_mols.parquet` | the 28 936 MSG query molecules (`inchikey_2d, smiles, formula, exact_mass, fold`). |
| `MassSpecGym1.5_retrieval_candidates_formula.json` | the current final formula candidates (uniform-trim reference). |
| `MassSpecGym1.5_retrieval_candidates_mass.json` | the current final mass candidates (uniform-trim reference). |
| `MassSpecGym1.5.tsv` | the MSG dataset itself (only needed for sanity-checks). |

---

## Full pipeline (for reference)

The complete pipeline runs in 11 stages. Each row's "Inputs" → "Outputs"
column documents the data flow.

| # | Stage | Script | Compute | Inputs → Outputs |
|---|---|---|---|---|
| 0 | MCES2 ≤ 2 filter against val | `build_mces2_valtest_disjoint_pool.py` | CPU, ~1 h | published MSG MCES2-test-disjoint 4 M pool → `data/pretrain_corpora/MassSpecGym_molecules_MCES2_disjoint_with_valtest_4M.tsv` (3.91 M) |
| 1 | Tokeniser + 95/5 pretrain-zip prep | `prep_msg4m_pretrain_zips.py` | CPU, ~30 min | MSG-disjoint 4 M + MSG-train SMILES → `*_train.zip`, `*_valid.zip` (auto-built tokenizer covers MSG-train tokens → 0 OOV at finetune) |
| 2 | TSV canonicalisation | `rdkit_canon_massspecgym.py` | CPU, ~5 min | `MassSpecGym1.5.tsv` raw → `MassSpecGym1.5.tsv` with rdkit-canonical SMILES + recomputed `inchikey, formula, parent_mass, precursor_mz` |
| 3 | PubChem JSON canonicalisation | `make_pubchem_cands_canonical.py` | CPU, ~30 min | published PubChem candidate JSONs → rdkit-canonical-SMILES variant (used as the PubChem-augment source) |
| 4 | S4 pretrain | `build_msg_s4_candidates.py --stage pretrain` | **1 × MI250x GCD, ~24 h** | `chembl_std_*.zip` (step 1 output, 95/5 split of MSG-4M ∪ MSG-train) → `ckpt_pretrain/` (30 epochs, val_loss 0.179) |
| 5 | S4 finetune on MSG-train | `build_msg_s4_candidates.py --stage finetune` | **1 × MI250x GCD, ~1 h** | `ckpt_pretrain/` + `msg_train.zip` → `ckpt_finetune_msg/` |
| 6 | S4 sampling (200 M, 3 temps) | `build_msg_s4_candidates.py --stage sample` | **24 × MI250x GCD via SLURM array, ~4 h** | `ckpt_finetune_msg/` → `samples/temp_{0.70,1.00,1.20}/shard_*.parquet` (3 × 66.7 M = 200 M raw SMILES) |
| 7 | Standardise + dedup pool | `build_msg_s4_candidates.py --stage bucket` | **CPU, 128 cores, ~1 h** | all `samples/temp_*/shard_*.parquet` → `buckets.parquet` (59.6 M deduped by 2D-InChIKey, keeping highest LL) |
| 8 | Emit per-query candidate JSONs | `build_msg_s4_candidates.py --stage emit_{formula,mass}_json` | CPU, ~30 min each | `buckets.parquet` + `extract_unique_mols.parquet` → `MassSpecGym_S4_retrieval_candidates_{formula,mass}_nostereo.json` (S4-only, cap=512, uniform-random trim — **this is where your similarity-weighted sampler plugs in**) |
| 9 | PubChem augment of sparse buckets | `augment_s4_with_pubchem.py` | CPU, ~1 h | S4-only JSONs + canonical PubChem JSONs → `MassSpecGym_S4plusPC_retrieval_candidates_{formula,mass}_nostereo.json` (fill buckets with `< 8` S4 candidates from PubChem; cap=512) |
| 10 | Molpher augment of remaining sparse buckets (formula only) | `augment_s4_with_molpher.py` | CPU, ~2 h | S4+PC formula JSON → `MassSpecGym_S4plusPCplusMol_retrieval_candidates_formula_nostereo.json` (Molpher morph buckets still < 8 after step 9; cap=512) |
| 11 | Final JSON-value canonicalisation | `canonicalise_v15_candidate_values.py` | CPU, ~15 min | S4+PC+Mol JSONs → `data/v1.5/MassSpecGym1.5_retrieval_candidates_{formula,mass}.json` (the files MSG actually loads) |

Stage 8 (formula or mass emit) is where the **uniform-random trim** lives
that the collaborator is replacing.

---

## How sampling/trimming currently flows

There are **3 trim points** in the pipeline. All currently use uniform-random.
A similarity-weighted variant could be applied at any/all of them. The
choice depends on whether you want the bias applied:

- **At stage 8** (`build_msg_s4_candidates.py` emit_*): biases the
  *S4-only* candidates → the augment stages still fall back to PubChem
  for sparse buckets, so this affects mostly the dense-bucket case
  (≥99 % of queries).
- **At stage 9** (`augment_s4_with_pubchem.py`): biases the *post-PC*
  list once S4 + PubChem are merged; runs only when bucket size > 8.
- **At stage 10** (`augment_s4_with_molpher.py`): biases the *post-Molpher*
  list (formula only); runs only when bucket size still < 8 after stages
  8+9.

The simplest, most-impactful change is **at stage 8** — that's the main
bucket-trim. Stages 9 and 10 only kick in on sparse buckets and use the
same `random.sample` pattern, so the same trim function can be swapped
in once and threaded through.

---

## Running the pipeline end-to-end

The full pipeline was orchestrated on LUMI via SLURM. Reference submit
scripts are in `slurm/` — they are **LUMI-specific** (paths, partition
`standard-g`, account `project_465002061`, AMD MI250x GCDs) and will
need adaptation. Stage map:

| Stage | SLURM script | Resource |
|---|---|---|
| 0 | `slurm/submit_mces2_val_filter.sh` | CPU, standard, ~1 h |
| 1, 4, 5 | `slurm/submit_s4v2_prep.sh`, `submit_s4v2_pretrain.sh`, `submit_s4v2_finetune.sh` | GPU, standard-g |
| 6 | `slurm/submit_s4v2_sample.sh` | SLURM array 0–23, 1 GPU each, standard-g |
| 7, 8 | `slurm/submit_s4v2_bucket_emit.sh` | CPU, 128 cores, standard |

Stages 9–11 (augments + final canon) were run interactively / via short
CPU jobs. To re-run only the emit + augment + canon path (which is all
the collaborator needs), bypass SLURM and call the Python scripts
directly with `--n-workers 64` or similar.

---

## Hardcoded paths to be aware of

Several scripts contain absolute paths to the LUMI workspace:

```python
DATA = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data")
WS   = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev")
```

In particular:

- `augment_s4_with_pubchem.py` and `augment_s4_with_molpher.py` define
  `DATA = .../MassSpecGym/data` as a module-level constant and reference
  fixed filenames. Adapt these constants (or refactor to argparse) before
  running off LUMI.
- `build_mces2_valtest_disjoint_pool.py` has `WS = .../DreaMS-Mol_dev`.
- `build_msg_s4_candidates.py` is the most parameterised — almost
  everything is on `argparse` (`--out-dir`, `--dataset-tsv`, `--s4-repo`,
  `--out-formula-json`, etc.).

For the collaborator's task (improving sampling), only
`build_msg_s4_candidates.py` and the two augment scripts matter, and only
their **trim function** + **call site** needs to change — the paths can
be left alone if running on LUMI, or globally substituted otherwise.

---

## Dependencies

- `rdkit` (≥ 2023.09) — canonicalisation, fingerprints
- `matchms` — MGF I/O (for the TSV canon step that recomputes `precursor_mz`)
- `s4dd` — the S4 language model
  (`https://github.com/molML/s4-for-de-novo-drug-design`, cloned alongside
  the workspace as `s4-for-de-novo-drug-design/`; passed via `--s4-repo`)
- `molpher-lib` — molecule-morphing (Molpher augment); **conda-only**
  package, install in a dedicated conda env
- `massspecgym` — utilities (`rdkit_canonical_smiles`,
  `_strip_stereo_parallel`, `MyopicMCES`). The pipeline imports these
  from the installed `massspecgym` package, not via relative path.
- `pyarrow`, `pandas`, `numpy`, `tqdm`

---

## What the collaborator delivers back

The minimal deliverable to plug into the v1.5 release:

1. **One modified file:** `build_msg_s4_candidates.py` with the new
   `_trim_*` function and its two call sites in `_stage_emit_formula_json`
   / `_stage_emit_mass_json` updated.
2. **Two updated JSONs:** the re-emitted
   `MassSpecGym_S4_retrieval_candidates_{formula,mass}_nostereo.json`.
3. **(Optionally)** the propagated changes through
   `augment_s4_with_pubchem.py` / `augment_s4_with_molpher.py` and the
   final canon'd `MassSpecGym1.5_retrieval_candidates_{formula,mass}.json`.

A self-contained PR to this branch is the cleanest hand-off.
