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
OUT  = DATA / "test_results_v1.5/retrieval"

TSV_NS = DATA / "MassSpecGym1.5_nostereo.tsv"
JSONS = {
    "formula": DATA / "MassSpecGym_S4plusPCplusMol_retrieval_candidates_formula_nostereo.json",
    "mass":    DATA / "MassSpecGym_S4plusPC_retrieval_candidates_mass_nostereo.json",
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
        # Per-spectrum list-size N_i.
        Ns = test["smiles"].map(lambda s: len(cands.get(s, []))).fillna(0).astype(int).values
        n_missing = int((Ns == 0).sum())
        if n_missing:
            print(f"  WARNING: {n_missing:,} test spectra had no candidates JSON entry")
        # Avoid /0 for missing entries.
        Ns_safe = np.where(Ns > 0, Ns, 1)

        # ---- random baseline -------------------------------------------------
        # Bernoulli draw per spectrum: hit@k = 1 if rng.random() < k/N_i.
        r1  = (rng.random(len(test)) < 1  / Ns_safe).astype(float)
        r5  = (rng.random(len(test)) < 5  / Ns_safe).astype(float)
        r20 = (rng.random(len(test)) < 20 / Ns_safe).astype(float)
        # Zero-out the ones we have no candidates for.
        r1[Ns == 0]  = 0.0
        r5[Ns == 0]  = 0.0
        r20[Ns == 0] = 0.0
        # Per-spectrum MRR for random: expected = (1/N) * H_N (harmonic).
        # Use a single realisation: rank ~ uniform in {1..N}.
        ranks = rng.integers(1, Ns_safe + 1)
        ranks = np.where(Ns > 0, ranks, 1)
        mrr = np.where(Ns > 0, 1.0 / ranks, 0.0)
        df_random = pd.DataFrame({
            "identifier": test["identifier"],
            "test_hit_rate@1":  r1,
            "test_hit_rate@5":  r5,
            "test_hit_rate@20": r20,
            "test_mrr":         mrr,
        })
        path = OUT / f"random_{variant}_nostereo.pkl"
        with path.open("wb") as f: pickle.dump(df_random, f)
        print(f"  random:    wrote {path.name}  (hit@1 = {df_random['test_hit_rate@1'].mean()*100:.3f}%)")

        # ---- chirality per-spectrum -----------------------------------------
        with CHIRPKLS[variant].open("rb") as f:
            chir = pickle.load(f)["df"]  # per-SMILES df with hit_rate@k cols
        chir_map = {row["smiles"]: row for _, row in chir.iterrows()}
        rows = []
        for _, r in test.iterrows():
            base = chir_map.get(r["smiles"])
            if base is None:
                rows.append({"identifier": r["identifier"], "test_hit_rate@1": 0.0,
                             "test_hit_rate@5": 0.0, "test_hit_rate@20": 0.0})
                continue
            rows.append({
                "identifier": r["identifier"],
                "test_hit_rate@1":  float(base["hit_rate@1"]),
                "test_hit_rate@5":  float(base["hit_rate@5"]),
                "test_hit_rate@20": float(base["hit_rate@20"]),
            })
        df_chir = pd.DataFrame(rows)
        path = OUT / f"chirality_{variant}_nostereo_per_spectrum.pkl"
        with path.open("wb") as f: pickle.dump(df_chir, f)
        print(f"  chirality: wrote {path.name}  (hit@1 = {df_chir['test_hit_rate@1'].mean()*100:.3f}%)")


if __name__ == "__main__":
    main()
