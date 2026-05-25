#!/usr/bin/env python
"""Add a ``test_mces@1`` column to per-spectrum baseline pickles.

For each test spectrum we compute MyopicMCES between the ground-truth
SMILES (from the TSV) and the *top-1* predicted candidate.

How top-1 is determined per ``--method``:
- ``sorted_smiles``: read ``sorted_candidate_smiles[0]`` directly from
  the pkl. Used for chemberta + learnable models.
- ``random``: uniform-random pick from the query's candidate list
  (excluding the query SMILES at index 0).
- ``chirality``: refit the chirality sort (training-fitted direction)
  and take the chirality-sorted top-1 with stable random tie-breaking
  (matches the per-spectrum logic in ``eval_chirality_baseline.py``).

The script writes a new pkl with the added column (or appends to the
existing pkl if ``--in_place``). Parallelised across spectra with a
worker pool.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from massspecgym.utils import MyopicMCES


# Worker globals (one MCES instance per worker)
_WORKER_MCES: Optional[MyopicMCES] = None


def _init_worker():
    global _WORKER_MCES
    _WORKER_MCES = MyopicMCES()


def _mces_pair(args):
    smi1, smi2 = args
    if not smi1 or not smi2:
        return float("nan")
    try:
        return float(_WORKER_MCES(smi1, smi2))
    except Exception:
        return float("nan")


def _chir_count(smi: str) -> int:
    """Count *potential* stereocenters from the 2D connectivity.

    Kept byte-identical to ``scripts/eval_chirality_baseline.py::chir_count``
    so the MCES top-1 derived here is consistent with the chirality
    baseline's hit-rate / MRR computation. After stereo-strip (v1.5), the
    annotated-chiral-atom count is 0 for every molecule, so we count
    potential stereocenters from the 2D graph instead.
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0
    try:
        return len(Chem.FindMolChiralCenters(
            m, includeUnassigned=True, useLegacyImplementation=False))
    except Exception:
        return 0


def _fit_chirality_direction(tsv: pd.DataFrame, cands: dict[str, list[str]],
                              chir_cache: dict[str, int]) -> str:
    """Same direction-fitting rule as scripts/eval_chirality_baseline.py."""
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
    return "desc" if mean_delta > 0 else "asc"


def _top1_random(query: str, cands: dict, rng: random.Random) -> Optional[str]:
    cand_list = cands.get(query, [])
    if len(cand_list) <= 1:
        return None
    # Random pick from non-query positions; matches the random baseline notion.
    return rng.choice(cand_list[1:]) if len(cand_list) > 1 else None


def _top1_chirality(query: str, cands: dict, chir_cache: dict, sign: float,
                    rng: random.Random) -> Optional[str]:
    cand_list = cands.get(query, [])
    if not cand_list:
        return None
    scores = np.asarray([chir_cache.get(c, 0) for c in cand_list], dtype=np.float32) * (-sign)
    # Stable argsort with tiny random jitter — same tie-break logic as
    # eval_chirality_baseline.py::hit_at_k_averaged but a single
    # realisation here (matches mean MCES@1 in expectation).
    jitter = rng.random()
    rng.seed(rng.randint(0, 1_000_000))  # mix per-query
    j = np.asarray([rng.random() for _ in scores], dtype=np.float32) * 1e-9
    order = np.argsort(-(scores + j), kind="stable")
    return cand_list[int(order[0])]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", type=Path, required=True)
    p.add_argument("--tsv", type=Path, required=True)
    p.add_argument("--cands", type=Path,
                   help="Required for method=random|chirality. Optional for sorted_smiles.")
    p.add_argument("--method", choices=["sorted_smiles", "random", "chirality"], required=True)
    p.add_argument("--n_workers", type=int, default=int(os.environ.get("WORKERS", mp.cpu_count())))
    p.add_argument("--out", type=Path, help="Output pkl path (default: overwrite input)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = args.out or args.pkl

    print(f"Loading pkl {args.pkl}")
    df = pd.read_pickle(args.pkl)
    print(f"  {len(df):,} rows; cols={list(df.columns)}")

    if "test_mces@1" in df.columns:
        print(f"  test_mces@1 already exists; will be overwritten")

    print(f"Loading TSV {args.tsv}")
    tsv = pd.read_csv(args.tsv, sep="\t", usecols=["identifier", "smiles", "fold"])
    id_to_smi = dict(zip(tsv["identifier"], tsv["smiles"]))

    # Derive top-1 per spectrum row.
    top1_smiles: list[Optional[str]] = []
    gt_smiles: list[str] = []
    if args.method == "sorted_smiles":
        if "sorted_candidate_smiles" not in df.columns:
            raise SystemExit("pkl has no sorted_candidate_smiles — wrong method")
        for r in df.itertuples(index=False):
            scs = r.sorted_candidate_smiles
            top1_smiles.append(scs[0] if (scs is not None and len(scs) > 0) else None)
            gt_smiles.append(id_to_smi.get(r.identifier, None))
    else:
        if args.cands is None:
            raise SystemExit("--cands is required for method=random|chirality")
        print(f"Loading cands {args.cands}")
        with args.cands.open() as f:
            cands = json.load(f)
        if args.method == "random":
            rng = random.Random(args.seed)
            for r in df.itertuples(index=False):
                q = id_to_smi.get(r.identifier)
                top1_smiles.append(_top1_random(q, cands, rng) if q else None)
                gt_smiles.append(q)
        else:  # chirality
            print("Computing chirality counts ...")
            unique_smi = set()
            for q, cs in cands.items():
                unique_smi.add(q)
                unique_smi.update(cs)
            unique_smi.update(tsv["smiles"].dropna().tolist())
            with mp.get_context("fork").Pool(args.n_workers) as pool:
                vals = pool.map(_chir_count, sorted(unique_smi), chunksize=2000)
            chir_cache = dict(zip(sorted(unique_smi), vals))
            direction = _fit_chirality_direction(tsv, cands, chir_cache)
            sign = -1.0 if direction == "desc" else 1.0
            print(f"  direction={direction}")
            rng = random.Random(args.seed)
            for r in df.itertuples(index=False):
                q = id_to_smi.get(r.identifier)
                top1_smiles.append(_top1_chirality(q, cands, chir_cache, sign, rng) if q else None)
                gt_smiles.append(q)

    # Compute MCES@1 in parallel.
    n_valid = sum(1 for t in top1_smiles if t)
    print(f"Computing MCES@1 for {n_valid:,}/{len(top1_smiles):,} spectra using {args.n_workers} workers ...")
    pairs = list(zip(gt_smiles, top1_smiles))
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(args.n_workers, initializer=_init_worker) as pool:
        # imap to preserve order with progress reporting.
        results = []
        for i, val in enumerate(pool.imap(_mces_pair, pairs, chunksize=8)):
            results.append(val)
            if (i + 1) % 1000 == 0 or i == len(pairs) - 1:
                el = time.perf_counter() - t0
                rate = (i + 1) / max(el, 0.001)
                eta = (len(pairs) - i - 1) / max(rate, 0.001)
                mean_so_far = float(np.nanmean(results)) if results else float("nan")
                print(f"  {i + 1:,}/{len(pairs):,}  rate={rate:.1f}/s  ETA={eta/60:.1f}m  mean_mces@1={mean_so_far:.3f}",
                      flush=True)

    df["test_mces@1"] = results
    n_nan = int(np.isnan(results).sum())
    print(f"\n  test_mces@1 mean = {np.nanmean(results):.3f}, nans = {n_nan}")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
