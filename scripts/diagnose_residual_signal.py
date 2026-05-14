#!/usr/bin/env python
"""Diagnose what residual signal remains in S4-augmented candidates after
stereo-stripping. For each simple SMILES feature (length, heavy_atoms,
ring_count, aromatic_atoms, heteroatom_count, has_unusual_element), score
candidates by the feature value (with fitted direction on train), report
hit@1 / hit@5 / hit@20. Compare distributions of each feature on MSG test
queries vs S4 candidate pool.

If any single feature reaches close to ChemBERTa's 1.5% hit@1 on the
nostereo data, that's an undiscovered shortcut.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


COMMON_ELEMENTS = {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"}


def features(smi: str) -> dict:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {k: 0 for k in [
            "length", "heavy_atoms", "n_rings", "aromatic_atoms",
            "heteroatoms", "n_unusual_elements", "max_ring_size", "n_nitrogens",
        ]}
    atoms = list(m.GetAtoms())
    n_heavy = sum(1 for a in atoms if a.GetAtomicNum() > 1)
    syms = [a.GetSymbol() for a in atoms]
    n_unusual = sum(1 for s in syms if s not in COMMON_ELEMENTS)
    n_arom = sum(1 for a in atoms if a.GetIsAromatic())
    n_het = sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6))
    n_n = sum(1 for s in syms if s == "N")
    ring_info = m.GetRingInfo()
    n_rings = ring_info.NumRings()
    max_ring = max((len(r) for r in ring_info.AtomRings()), default=0)
    return {
        "length": len(smi),
        "heavy_atoms": n_heavy,
        "n_rings": n_rings,
        "aromatic_atoms": n_arom,
        "heteroatoms": n_het,
        "n_unusual_elements": n_unusual,
        "max_ring_size": max_ring,
        "n_nitrogens": n_n,
    }


def _worker_init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _worker_features(smi):
    return smi, features(smi)


def _features_in_pool(smiles, n_workers):
    items = list(smiles)
    if n_workers <= 1:
        return dict(_worker_features(s) for s in items)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        return dict(pool.imap_unordered(_worker_features, items, chunksize=2000))


def hit_at_k(scores, gt_idx, k, rng):
    jitter = rng.random(len(scores)) * 1e-9
    order = np.argsort(-(scores + jitter), kind="stable")
    return int(np.where(order == gt_idx)[0][0] < k)


def fit_direction(tsv, cands, feat_cache, feature_key):
    train_smiles = sorted(tsv[tsv["fold"] == "train"]["smiles"].dropna().unique().tolist())
    deltas = []
    for smi in train_smiles:
        if smi not in cands:
            continue
        cand_list = cands[smi]
        if len(cand_list) < 2:
            continue
        v_gt = feat_cache.get(smi, {}).get(feature_key, 0)
        decoy_vs = [feat_cache.get(c, {}).get(feature_key, 0) for c in cand_list[1:]]
        if not decoy_vs:
            continue
        deltas.append(v_gt - float(np.mean(decoy_vs)))
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    direction = "desc" if mean_delta > 0 else "asc"
    return direction, mean_delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True, type=Path)
    p.add_argument("--cands", required=True, type=Path)
    p.add_argument("--out_pkl", required=True, type=Path)
    p.add_argument("--folds", default="test")
    p.add_argument("--n_workers", type=int, default=int(os.environ.get("WORKERS", 32)))
    args = p.parse_args()

    print(f"Loading TSV {args.tsv}")
    tsv = pd.read_csv(args.tsv, sep="\t", usecols=["smiles", "fold"])
    print(f"  {len(tsv):,} rows")

    print(f"Loading cands {args.cands}")
    t0 = time.perf_counter()
    with args.cands.open() as f:
        cands = json.load(f)
    print(f"  {len(cands):,} keys in {time.perf_counter()-t0:.1f}s")

    smi_set = set(cands.keys())
    for v in cands.values():
        smi_set.update(v)
    smi_set.update(tsv["smiles"].dropna().tolist())
    print(f"  {len(smi_set):,} unique SMILES (queries + candidates)")
    t0 = time.perf_counter()
    feat_cache = _features_in_pool(sorted(smi_set), args.n_workers)
    print(f"  features computed in {time.perf_counter()-t0:.1f}s")

    folds = set(args.folds.split(","))
    target_smiles = tsv[tsv["fold"].isin(folds)]["smiles"].dropna().unique().tolist()
    target_smiles = [s for s in target_smiles if s in cands]
    print(f"  scoring {len(target_smiles):,} unique {sorted(folds)} queries")

    # Distribution comparison: queries vs all candidates.
    feature_keys = list(features("C").keys())
    print("\n=== Distribution: MSG test queries vs S4 candidate pool ===")
    rng_dist = np.random.RandomState(0)
    pool_smi = rng_dist.choice(list(smi_set - set(target_smiles)), size=min(50000, len(smi_set)), replace=False)
    dist_rows = []
    for k in feature_keys:
        gt_vals = np.asarray([feat_cache.get(s, {}).get(k, 0) for s in target_smiles], dtype=float)
        cand_vals = np.asarray([feat_cache.get(s, {}).get(k, 0) for s in pool_smi], dtype=float)
        dist_rows.append({
            "feature": k,
            "GT mean": float(gt_vals.mean()), "GT median": float(np.median(gt_vals)), "GT p10": float(np.percentile(gt_vals, 10)), "GT p90": float(np.percentile(gt_vals, 90)),
            "POOL mean": float(cand_vals.mean()), "POOL median": float(np.median(cand_vals)), "POOL p10": float(np.percentile(cand_vals, 10)), "POOL p90": float(np.percentile(cand_vals, 90)),
            "Δ mean (GT-POOL)": float(gt_vals.mean() - cand_vals.mean()),
        })
    df_dist = pd.DataFrame(dist_rows)
    print(df_dist.to_string(index=False))

    # Per-feature hit@k baseline.
    print("\n=== Per-feature ranking baseline (fitted direction on train) ===")
    rng = np.random.default_rng(0)
    perf_rows = []
    for key in feature_keys:
        direction, mean_delta = fit_direction(tsv, cands, feat_cache, key)
        sign = -1.0 if direction == "desc" else 1.0
        n_hit1 = n_hit5 = n_hit20 = 0
        n = 0
        for q in target_smiles:
            cand_list = cands[q]
            if not cand_list:
                continue
            s_arr = np.asarray([feat_cache.get(c, {}).get(key, 0) for c in cand_list], dtype=np.float32) * (-sign)
            gt_idx = 0
            n += 1
            n_hit1  += hit_at_k(s_arr, gt_idx, 1, rng)
            n_hit5  += hit_at_k(s_arr, gt_idx, 5, rng)
            n_hit20 += hit_at_k(s_arr, gt_idx, 20, rng)
        perf_rows.append({
            "feature": key, "direction": direction, "mean_delta": mean_delta,
            "hit@1 (%)": 100*n_hit1/max(n,1), "hit@5 (%)": 100*n_hit5/max(n,1), "hit@20 (%)": 100*n_hit20/max(n,1),
            "n_queries": n,
        })
    df_perf = pd.DataFrame(perf_rows)
    df_perf = df_perf.sort_values("hit@1 (%)", ascending=False)
    print(df_perf.to_string(index=False))

    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open("wb") as f:
        pickle.dump({"dist": df_dist, "perf": df_perf}, f)
    print(f"\nWrote {args.out_pkl}")


if __name__ == "__main__":
    main()
