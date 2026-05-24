# `massspecgym_in_the_wild` — notebooks index

The four notebooks tracked in this folder analyse the published v1.5 release.

| Notebook | Purpose |
|---|---|
| `v1.5_consistency_check.ipynb` | Row-by-row diff of `MassSpecGym1.5.tsv` / `.mgf` vs the original PubChem-standardised release. Verifies that every diff is confined to columns the canon script recomputes (`smiles`, `formula`, `inchikey`, `parent_mass`, `precursor_formula`) and that MGF peak lists are byte-identical. |
| `candidates_analysis_final.ipynb` | Full analysis of the v1.5 candidate JSONs: per-step source attribution (S4 / PubChem / Molpher), per-formula and per-mass bucket distributions, qualitative examples, pool quality, internal Tanimoto. Reads the v1.5 finals plus intermediate stage outputs in `data/` to compute the per-step breakdown. |
| `evaluation_s4.ipynb` | Retrieval baselines (random, chirality, ChemBERTa, DeepSets, DeepSets+FF, FingerprintFFN) evaluated against the v1.5 candidate JSONs. Reports hit@1 / hit@5 / hit@20 / MRR / MCES@1 with 99.9 % CIs on both mass and formula test sets. |
| `candidates_construction_S4.ipynb` | Diagnostics on the 64.2 M-mol S4 pool that underlies the candidate JSONs (temperature trade-offs, formula-bucket coverage, validity). Documents how the candidates were built. |

Workspace-only files (helper `.py` scripts, generated figure dirs,
research notebooks like `papers.ipynb`, `incorrect_metrics_de_novo.ipynb`)
are intentionally not tracked in git.
