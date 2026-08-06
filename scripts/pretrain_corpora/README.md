# Pretraining molecule corpora

Code that builds and verifies the unlabeled molecule corpora used to pretrain generative and
de-novo models. The corpora themselves live at
`${DREAMSMOL_DATA}/MassSpecGym/pretrain_corpora/` (not in the repo).

## The corpus files, and which one is canonical

| File | Molecules | Rule | Status |
|---|---|---|---|
| `MassSpecGym_retrieval_molecules_4M.tsv` | 4,175,091 | none — the unfiltered parent set | reference |
| `MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv` | 3,922,527 | MassSpecGym's own: MCES **< 2** vs **test**, plus 2D-InChIKey dedup vs all three folds | published on Hugging Face; sha256 `6f9e07ac…` |
| **`MassSpecGym_molecules_MCES2_disjoint_with_valtest_fold_4M.tsv`** | see `build_summary.json` | published pool minus MCES **< 2** vs **val ∪ test**, exact | **CANONICAL** |
| `MassSpecGym_molecules_MCES2_disjoint_with_valtest_4M_legacy_withHprefilter.tsv` | 3,914,223 | published pool minus MCES **≤ 2** vs val, **hydrogen-inclusive formula pre-filter** | **SUPERSEDED — do not train on this** |

Each file is a pure, order-preserving row deletion of the one above it.

### Why the legacy file was superseded

It has two defects, pulling in opposite directions
(audit: `experiments/reports/data-eda/2026-08-06_pretrain-4m-mces2-audit/`):

|  | **d < 2** (near-duplicates) | **d = 2 exactly** |
|---|---|---|
| pairs it looked at (with-H formula L1 ≤ 2) | removed ✅ | removed ✅ |
| its blind spot (heavy L1 ≤ 2 but with-H L1 > 2) | **kept** ❌ | kept |

MCES runs on the **heavy-atom** graph — `myopic_mces.graph.construct_graph` never calls `AddHs` —
so screening candidate pairs on the hydrogen-inclusive formula silently skipped CH2 homologs
(d = 1, with-H L1 = 3) and CH3↔Cl swaps (d = 2, with-H L1 = 5). It was stricter where it looked and
blind everywhere else.

## Scripts

| Script | Purpose |
|---|---|
| `build_mces_lt2_valtest_pool.py` | builds the canonical corpus (`prep` → `score` array → `assemble`) |
| `verify_mces2_disjointness.py` | audits any pool against any fold; `--selftest` checks MCES semantics |
| `slurm/submit_mces_lt2_build.sh` | Karolina `qcpu_free` driver for the three build stages |
| `slurm/submit_mces2_audit.sh` | Karolina driver for the audit probes |

The old builder `../v1.5_candidates/build_mces2_valtest_disjoint_pool.py` stays where it is — it is
genuinely stage 0 of the v1.5 candidate pipeline — and carries a docstring recording both defects.

## The candidate screen is exact, not a heuristic

`build_mces_lt2_valtest_pool.py` does not compare all 3,922,527 × 6,181 ≈ 24 billion pairs. It
scores only pairs whose **heavy-atom formula L1 distance ≤ 2**, which provably cannot drop a hit.
From the ILP objective in `myopic_mces/MCES_ILP.py`:

```
cost = Σ w(unmapped edges) + Σ |w1−w2| (mapped edge pairs),   min bond weight = 1.0
```

- `MCES < 2` ⟹ at most **one** unmapped bond across both graphs.
- An unmapped atom must have all its bonds unmapped (ILP: *"the mapping of the edges has to match
  the mapping of the nodes"*).
- Both graphs are connected — asserted at build time; the build **aborts** on any `.`-containing
  SMILES, because a disconnected zero-degree atom costs nothing in the objective and would break
  the bound.
- ⟹ at most one unmapped atom ⟹ heavy-atom formula L1 ≤ **1**. We screen at 2 for margin.

## Two MCES gotchas worth knowing

1. **The threshold sentinel.** `MCES_ILP` constrains the objective to `<= threshold` and returns
   `threshold` itself when the problem is infeasible, so at `threshold=2` a returned `2.0` means
   *either* "exactly 2" *or* "> 2, unproven". It is harmless for a strict `< 2` rule (neither is
   `< 2`) but it makes `<= 2` rules over-remove, and it makes "we found N pairs at exactly 2"
   wrong unless you rescore at a higher threshold.
2. **No `timeLimit`.** MassSpecGym's `MyopicMCES` sets none. Adding one lets CBC stop early and
   return the same `threshold` sentinel, which reads as "≥ 2" and silently **under-removes**.
