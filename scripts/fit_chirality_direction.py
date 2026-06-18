"""Fit the direction (desc/asc) for the GT-free chirality random baseline.

For each (feature, pool) combo, computes the mean per-query offset
    Delta = n(GT_q) - mean(n(decoy_c) for c in candidates_q)
over the *training* split. If Delta > 0 the GT positives carry more
chirality than their decoys on average, so descending in n(c) (high first)
ranks positives near the top; otherwise ascending.

Output: data/test_results_v1.5/retrieval/chirality_directions.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TSV_PTH = DATA_DIR / "MassSpecGym_RDKit_SMILES.tsv"
POOLS = {
    "mass": DATA_DIR / "MassSpecGym_retrieval_candidates_mass_RDKit_SMILES.json",
    "formula": DATA_DIR / "MassSpecGym_retrieval_candidates_formula_RDKit_SMILES.json",
}
FEATURES = ("n_chir", "n_pot")
OUT_PTH = DATA_DIR / "test_results_v1.5" / "retrieval" / "chirality_directions.json"

_FEAT_CACHE: dict[tuple[str, str], int] = {}


def chir_count(smi: str, feature: str) -> int | None:
    key = (smi, feature)
    if key in _FEAT_CACHE:
        return _FEAT_CACHE[key]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        _FEAT_CACHE[key] = None
        return None
    if feature == "n_chir":
        n = sum(
            1 for a in mol.GetAtoms()
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        )
    else:  # n_pot
        try:
            n = len(Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, useLegacyImplementation=False
            ))
        except RuntimeError:
            try:
                n = len(Chem.FindMolChiralCenters(
                    mol, includeUnassigned=True, useLegacyImplementation=True
                ))
            except RuntimeError:
                n = sum(
                    1 for a in mol.GetAtoms()
                    if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                )
    _FEAT_CACHE[key] = n
    return n


def main() -> None:
    print(f"Loading TSV {TSV_PTH}", flush=True)
    tsv = pd.read_csv(TSV_PTH, sep="\t", usecols=["smiles", "fold"])
    train_smiles = sorted(tsv[tsv["fold"] == "train"]["smiles"].dropna().unique().tolist())
    print(f"  train queries: {len(train_smiles)}", flush=True)

    results: dict[str, dict[str, dict]] = {}
    for pool_name, pool_pth in POOLS.items():
        print(f"\nLoading pool {pool_name} from {pool_pth.name}", flush=True)
        t0 = time.time()
        with open(pool_pth) as f:
            cands = json.load(f)
        print(f"  loaded {len(cands)} sets in {time.time()-t0:.0f}s", flush=True)

        for feature in FEATURES:
            t0 = time.time()
            deltas: list[float] = []
            for q in train_smiles:
                if q not in cands:
                    continue
                lst = cands[q]
                if len(lst) < 2:
                    continue
                n_q = chir_count(q, feature)
                if n_q is None:
                    continue
                neg_counts = [chir_count(s, feature) for s in lst[1:]]
                neg_counts = [n for n in neg_counts if n is not None]
                if not neg_counts:
                    continue
                deltas.append(float(n_q) - float(np.mean(neg_counts)))
            delta_mean = float(np.mean(deltas)) if deltas else 0.0
            direction = "desc" if delta_mean > 0 else "asc"
            print(
                f"  {feature:>6s} × {pool_name:>7s}: "
                f"Δ̄ = {delta_mean:+.4f}  →  direction = {direction}  "
                f"(n_queries = {len(deltas)}, took {time.time()-t0:.0f}s)",
                flush=True,
            )
            results.setdefault(feature, {})[pool_name] = {
                "direction": direction,
                "delta": delta_mean,
                "n_queries": len(deltas),
            }

    OUT_PTH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PTH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PTH}")


if __name__ == "__main__":
    main()
