#!/usr/bin/env python
"""GT-free chirality retrieval baseline.

For each candidate SMILES, score by chiral-atom count
(``Chem.ChiralType.CHI_UNSPECIFIED`` exclusions). Direction (descending /
ascending) is chosen by fitting on the *train* split: take the average
``n(GT) - mean(n(decoys))`` over train queries; if positive, rank descending
(more chirality first), else ascending.

Inputs (CLI overrides):
  --tsv      MSG TSV with 'smiles', 'inchikey', 'fold' columns
  --cands    candidates JSON
  --out_pkl  output pickle with per-test-query DataFrame of hit_rate@{1,5,20}
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


def chir_count(smi: str) -> int:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0
    return sum(
        1 for a in m.GetAtoms()
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )


def _precompute_chir_counts(unique_smiles: list[str], n_workers: int) -> dict[str, int]:
    if n_workers <= 1:
        return {s: chir_count(s) for s in unique_smiles}
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        vals = pool.map(chir_count, unique_smiles, chunksize=2000)
    return dict(zip(unique_smiles, vals))


def fit_direction(tsv: pd.DataFrame, cands: dict[str, list[str]], chir_cache: dict[str, int]) -> str:
    """Fit direction on train queries using a pre-populated chir_count cache."""
    train_smiles = sorted(tsv[tsv["fold"] == "train"]["smiles"].dropna().unique().tolist())
    deltas = []
    for smi in train_smiles:
        if smi not in cands:
            continue
        cand_list = cands[smi]
        if len(cand_list) < 2:
            continue
        n_gt = chir_cache.get(smi, 0)
        decoy_ns = [chir_cache.get(c, 0) for c in cand_list[1:]]
        if not decoy_ns:
            continue
        deltas.append(n_gt - float(np.mean(decoy_ns)))
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    direction = "desc" if mean_delta > 0 else "asc"
    print(f"  train queries used: {len(deltas):,}  mean(Δ) = {mean_delta:.3f}  direction = {direction}")
    return direction


def hit_at_k_and_mrr_averaged(scores: np.ndarray, gt_idx: int, ks: tuple[int, ...],
                                rng: np.random.Generator, n_tie_breaks: int = 100,
                                ) -> tuple[tuple[float, ...], float]:
    """Return ((hit@k1, hit@k2, ...), mrr) averaged over ``n_tie_breaks``
    independent random tie-break orderings.

    Stable ``np.argsort`` + GT always at index 0 would give 100% hit@k when
    scores are all tied. A single random jitter realization fixes the
    spurious 100% but is itself a high-variance single draw (with most
    chirality counts being 0 after stereo-stripping, almost all candidates
    are tied — the result depends almost entirely on the jitter). Averaging
    over many tie-break realizations matches the random baseline's stability
    and gives the *expected* hit@k / MRR under uniform tie-breaking.
    """
    # (n_tie_breaks, len(scores))
    jitters = rng.random((n_tie_breaks, len(scores))) * 1e-9
    # Order indices per realization.
    orders = np.argsort(-(scores[None, :] + jitters), axis=1, kind="stable")
    # GT's rank (1-indexed) per realization.
    gt_ranks = (orders == gt_idx).argmax(axis=1) + 1
    hits = tuple(float((gt_ranks <= k).mean()) for k in ks)
    mrr = float((1.0 / gt_ranks).mean())
    return hits, mrr


def hit_at_k_averaged(scores: np.ndarray, gt_idx: int, ks: tuple[int, ...],
                       rng: np.random.Generator, n_tie_breaks: int = 100) -> tuple[float, ...]:
    """Back-compat wrapper. Prefer ``hit_at_k_and_mrr_averaged``."""
    hits, _ = hit_at_k_and_mrr_averaged(scores, gt_idx, ks, rng, n_tie_breaks)
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True, type=Path)
    p.add_argument("--cands", required=True, type=Path)
    p.add_argument("--out_pkl", required=True, type=Path)
    p.add_argument("--folds", default="test", help="comma-separated folds to score, e.g. 'val,test'")
    args = p.parse_args()

    print(f"Loading TSV {args.tsv}")
    tsv = pd.read_csv(args.tsv, sep="\t", usecols=["smiles", "inchikey", "fold"])
    print(f"  {len(tsv):,} rows")

    print(f"Loading cands {args.cands}")
    t0 = time.perf_counter()
    with args.cands.open() as f:
        cands = json.load(f)
    print(f"  {len(cands):,} keys in {time.perf_counter()-t0:.1f}s")

    # Precompute chir_count for ALL unique SMILES (train queries + all candidates) in parallel.
    print("Collecting unique SMILES ...")
    unique_smi: set[str] = set()
    for q, cand_list in cands.items():
        unique_smi.add(q)
        unique_smi.update(cand_list)
    unique_smi.update(tsv["smiles"].dropna().tolist())
    print(f"  {len(unique_smi):,} unique SMILES")

    n_workers = int(os.environ.get("WORKERS", multiprocessing.cpu_count()))
    print(f"  computing chir_count with {n_workers} workers ...")
    t0 = time.perf_counter()
    chir_cache = _precompute_chir_counts(sorted(unique_smi), n_workers)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    print("Fitting chirality direction on train fold ...")
    direction = fit_direction(tsv, cands, chir_cache)
    sign = -1.0 if direction == "desc" else 1.0

    folds = set(args.folds.split(","))
    target_smiles = tsv[tsv["fold"].isin(folds)]["smiles"].dropna().unique().tolist()
    target_smiles = [s for s in target_smiles if s in cands]
    print(f"Evaluating on {len(target_smiles):,} unique {sorted(folds)} queries")

    rows = []
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    for i, q in enumerate(target_smiles):
        cand_list = cands[q]
        if not cand_list:
            continue
        s_arr = np.asarray([chir_cache.get(c, 0) for c in cand_list], dtype=np.float32) * (-sign)
        gt_idx = 0
        (h1, h5, h20), mrr = hit_at_k_and_mrr_averaged(
            s_arr, gt_idx, (1, 5, 20), rng, n_tie_breaks=100)
        rows.append({
            "smiles": q,
            "n_cands": len(cand_list),
            "hit_rate@1":  h1,
            "hit_rate@5":  h5,
            "hit_rate@20": h20,
            "mrr":         mrr,
            "chir_count_gt": int(chir_cache.get(q, 0)),
        })
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(target_smiles)} processed in {time.perf_counter()-t0:.1f}s")

    df = pd.DataFrame(rows)
    print()
    print(f"hit_rate@1  = {df['hit_rate@1'].mean()*100:.2f}%")
    print(f"hit_rate@5  = {df['hit_rate@5'].mean()*100:.2f}%")
    print(f"hit_rate@20 = {df['hit_rate@20'].mean()*100:.2f}%")
    print(f"mrr         = {df['mrr'].mean()*100:.2f}%")
    print(f"direction: {direction}")
    print(f"n queries: {len(df):,}")

    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open("wb") as f:
        pickle.dump({"df": df, "direction": direction, "tsv": str(args.tsv), "cands": str(args.cands)}, f)
    print(f"Wrote {args.out_pkl}")


if __name__ == "__main__":
    main()
