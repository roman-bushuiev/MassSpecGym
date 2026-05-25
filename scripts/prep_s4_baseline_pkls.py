#!/usr/bin/env python
"""Prepare per-spectrum result pickles for the S4-augmented retrieval
baselines, in the format that `notebooks/massspecgym_in_the_wild/evaluation.ipynb`
expects (columns: identifier, test_hit_rate@{1,5,20}, optionally test_mrr).

Produces (all under ``data/test_results_v1.5/retrieval/``):
  random_mass_nostereo.pkl
  random_formula_nostereo.pkl
  chirality_mass_nostereo_per_spectrum.pkl
  chirality_formula_nostereo_per_spectrum.pkl

ChemBERTa per-spectrum pickles already exist with the correct schema.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data")
V15  = DATA / "v1.5"
OUT  = DATA / "test_results_v1.5/retrieval"

# Read from the published v1.5 files (canon TSV/MGF + canon-candidates JSONs).
TSV_NS = V15 / "MassSpecGym1.5.tsv"
JSONS = {
    "formula": V15 / "MassSpecGym1.5_retrieval_candidates_formula.json",
    "mass":    V15 / "MassSpecGym1.5_retrieval_candidates_mass.json",
}
CHIRPKLS = {
    "formula": OUT / "chirality_formula_nostereo.pkl",
    "mass":    OUT / "chirality_mass_nostereo.pkl",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Loading nostereo TSV {TSV_NS}")
    tsv = pd.read_csv(TSV_NS, sep="\t", usecols=["identifier", "smiles", "fold"])
    test = tsv[tsv["fold"] == "test"].reset_index(drop=True)
    print(f"  {len(test):,} test-fold spectrum rows; {test['smiles'].nunique():,} unique test SMILES")

    rng = np.random.default_rng(args.seed)
    for variant in ("formula", "mass"):
        print(f"\n=== {variant} ===")
        with JSONS[variant].open() as f:
            cands = json.load(f)

        # ---- random baseline ------------------------------------------------
        # Per Roman: "uniformly sample a random candidate for each query".
        # A single draw per query is high variance with 2,997 unique test
        # SMILES (small-N queries with many spectra can swing the per-spectrum
        # mean by ~0.5 percentage points across seeds). We average over
        # ``N_DRAWS`` random rank draws per query so the baseline reflects the
        # *expected* random performance rather than one noisy realisation.
        # Equivalent to the analytical k / N_i, with finite-sample noise that
        # vanishes as N_DRAWS grows.
        N_DRAWS = 100
        unique_smi = sorted(test["smiles"].dropna().unique().tolist())
        per_smi_N = np.asarray([len(cands.get(s, [])) for s in unique_smi], dtype=int)
        per_smi_safe = np.where(per_smi_N > 0, per_smi_N, 1)
        # (N_DRAWS, n_queries) matrix of uniform ranks in {1..N_i}.
        ranks_mat = rng.integers(1, per_smi_safe[None, :] + 1, size=(N_DRAWS, len(unique_smi)))
        per_smi_r1  = (ranks_mat <= 1 ).mean(axis=0).astype(float)
        per_smi_r5  = (ranks_mat <= 5 ).mean(axis=0).astype(float)
        per_smi_r20 = (ranks_mat <= 20).mean(axis=0).astype(float)
        per_smi_mrr = (1.0 / ranks_mat.astype(float)).mean(axis=0)
        per_smi_r1[per_smi_N == 0]  = 0.0
        per_smi_r5[per_smi_N == 0]  = 0.0
        per_smi_r20[per_smi_N == 0] = 0.0
        per_smi_mrr[per_smi_N == 0] = 0.0
        smi_to_row = dict(zip(unique_smi, zip(per_smi_r1, per_smi_r5, per_smi_r20, per_smi_mrr)))

        rows = []
        for _, r in test.iterrows():
            base = smi_to_row.get(r["smiles"], (0.0, 0.0, 0.0, 0.0))
            rows.append({
                "identifier": r["identifier"],
                "test_hit_rate@1":  base[0],
                "test_hit_rate@5":  base[1],
                "test_hit_rate@20": base[2],
                "test_mrr":         base[3],
            })
        df_random = pd.DataFrame(rows)
        path = OUT / f"random_{variant}_nostereo.pkl"
        # Preserve test_mces@1 if already computed by scripts/add_mces_at_1.py.
        if path.exists():
            try:
                prev = pd.read_pickle(path)
                if "test_mces@1" in prev.columns:
                    df_random = df_random.merge(
                        prev[["identifier", "test_mces@1"]], on="identifier", how="left")
            except Exception:
                pass
        with path.open("wb") as f: pickle.dump(df_random, f)
        print(f"  random:    wrote {path.name}  (hit@1 = {df_random['test_hit_rate@1'].mean()*100:.3f}%, sampled per SMILES then expanded)")

        # ---- chirality per-spectrum -----------------------------------------
        with CHIRPKLS[variant].open("rb") as f:
            chir = pickle.load(f)["df"]  # per-SMILES df with hit_rate@k cols
        chir_map = {row["smiles"]: row for _, row in chir.iterrows()}
        rows = []
        for _, r in test.iterrows():
            base = chir_map.get(r["smiles"])
            if base is None:
                rows.append({"identifier": r["identifier"], "test_hit_rate@1": 0.0,
                             "test_hit_rate@5": 0.0, "test_hit_rate@20": 0.0,
                             "test_mrr": 0.0})
                continue
            rows.append({
                "identifier": r["identifier"],
                "test_hit_rate@1":  float(base["hit_rate@1"]),
                "test_hit_rate@5":  float(base["hit_rate@5"]),
                "test_hit_rate@20": float(base["hit_rate@20"]),
                "test_mrr":         float(base.get("mrr", 0.0)),
            })
        df_chir = pd.DataFrame(rows)
        path = OUT / f"chirality_{variant}_nostereo_per_spectrum.pkl"
        # Preserve test_mces@1 if already computed by scripts/add_mces_at_1.py.
        if path.exists():
            try:
                prev = pd.read_pickle(path)
                if "test_mces@1" in prev.columns:
                    df_chir = df_chir.merge(
                        prev[["identifier", "test_mces@1"]], on="identifier", how="left")
            except Exception:
                pass
        with path.open("wb") as f: pickle.dump(df_chir, f)
        print(f"  chirality: wrote {path.name}  (hit@1 = {df_chir['test_hit_rate@1'].mean()*100:.3f}%)")


if __name__ == "__main__":
    main()
