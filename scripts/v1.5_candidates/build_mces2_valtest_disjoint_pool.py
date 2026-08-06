"""Filter the MCES2-test-disjoint 4M pool to also be MCES2-disjoint with MSG-val.

Optimized version: builds a dense numpy matrix of (formula_id × element_count),
then uses vectorised L1-distance scan to find near-formula matches in O(n) per
val molecule.

TWO KNOWN DEFECTS, both quantified by `verify_mces2_disjointness.py` (audit
2026-08-06, `experiments/reports/data-eda/2026-08-06_pretrain-4m-mces2-audit/`):

1. **The comparison sign does not match MassSpecGym's.** This script removes
   `d <= 2` (line ~158); MassSpecGym's published pool removes `d < 2` from the
   test fold (NeurIPS 2024 §2.6, confirmed by measurement: 1,733 surviving
   test pairs at exactly d = 2, none below). The resulting corpus is therefore
   cleaned one shell more strictly against val than against test.

2. **The candidate pre-filter is incomplete.** `formula_counter` calls
   `Chem.AddHs`, so the L1 <= max_delta screen below is over *hydrogen-inclusive*
   formulas — but MCES runs on the heavy-atom graph, where `d <= 2` only implies
   a *heavy-atom* L1 <= 2. The with-H screen is a strict subset of that bound, so
   pairs such as a CH2 homolog (d = 1, with-H L1 = 3) or a CH3<->Cl swap
   (d = 2, with-H L1 = 5) are never scored. Use `--prefilter heavy` semantics
   (heavy-atom formula) if rebuilding.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, get_context
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

WS = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev")
sys.path.insert(0, str(WS / "MassSpecGym"))


def formula_counter(smi: str):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return None
        m = Chem.AddHs(m)
        c = Counter()
        for a in m.GetAtoms():
            c[a.GetSymbol()] += 1
        return tuple(sorted(c.items()))
    except Exception:
        return None


def _mces_worker_init():
    from massspecgym.utils import MyopicMCES
    global _MCES
    # timeLimit=10s prevents single hard MCS instances from blocking workers.
    # If CBC hits the limit, it returns its best-so-far bound; MyopicMCES
    # treats that as a (possibly conservative) upper bound on MCES.
    _MCES = MyopicMCES(
        threshold=2,
        always_stronger_bound=True,
        solver_options={"msg": 0, "timeLimit": 10},
    )


def _mces_pair(args):
    v_smi, p_smi, p_idx = args
    try:
        d = _MCES(v_smi, p_smi)
        return (p_idx, d)
    except Exception:
        # Conservative: assume MCES > 2 on failure (keeps the pool molecule).
        return (p_idx, float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--msg-tsv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--max-delta", type=int, default=2)
    args = ap.parse_args()

    print(f"Loading pool {args.pool} ...")
    pool = pd.read_csv(args.pool, sep="\t")
    print(f"  pool: {len(pool):,} rows; cols={list(pool.columns)}")

    print(f"Loading MSG TSV {args.msg_tsv} ...")
    msg = pd.read_csv(args.msg_tsv, sep="\t", usecols=["smiles", "fold"])
    val_smiles = sorted(msg[msg.fold == "val"]["smiles"].dropna().unique().tolist())
    print(f"  val: {len(val_smiles):,} unique")

    pool_smiles = pool["smiles"].tolist()

    print(f"\nComputing formulas (parallel, {args.workers} workers) ...")
    t0 = time.perf_counter()
    with Pool(args.workers) as p:
        pool_keys = p.map(formula_counter, pool_smiles, chunksize=4000)
        val_keys  = p.map(formula_counter, val_smiles,  chunksize=200)
    print(f"  formulas done in {time.perf_counter()-t0:.1f}s")

    # Build unique-formula table + element-count matrix.
    unique_formula = list({k for k in pool_keys if k is not None})
    print(f"  unique pool formulas: {len(unique_formula):,}")

    # Discover all elements present
    elem_set = set()
    for k in unique_formula:
        for e, _ in k:
            elem_set.add(e)
    for k in val_keys:
        if k is None: continue
        for e, _ in k:
            elem_set.add(e)
    elements = sorted(elem_set)
    print(f"  elements seen: {elements}")

    def to_vec(k):
        v = np.zeros(len(elements), dtype=np.int32)
        if k is None: return v
        d = dict(k)
        for i, e in enumerate(elements):
            v[i] = d.get(e, 0)
        return v

    print(f"\nBuilding dense formula matrix ({len(unique_formula)} × {len(elements)}) ...")
    t0 = time.perf_counter()
    F_pool = np.stack([to_vec(k) for k in unique_formula])  # (n_uniq, n_elem)
    F_val  = np.stack([to_vec(k) for k in val_keys])         # (n_val, n_elem)
    print(f"  done in {time.perf_counter()-t0:.1f}s; F_pool shape={F_pool.shape}, F_val shape={F_val.shape}")

    # Map formula -> list of pool indices
    pool_by_formula: dict = defaultdict(list)
    for idx, k in enumerate(pool_keys):
        if k is not None:
            pool_by_formula[k].append(idx)

    # For each val mol, find unique-formula indices within max-delta L1, then expand to pool indices.
    print(f"\nMatching val mols → near-formula pool indices (vectorised, max Δ={args.max_delta}) ...")
    t0 = time.perf_counter()
    pairs = []
    for vi in range(len(val_smiles)):
        v_vec = F_val[vi]
        if v_vec.sum() == 0:
            continue
        deltas = np.abs(F_pool - v_vec[None, :]).sum(axis=1)
        near_idx = np.where(deltas <= args.max_delta)[0]
        v_smi = val_smiles[vi]
        for fi in near_idx:
            k = unique_formula[fi]
            for p_idx in pool_by_formula[k]:
                pairs.append((v_smi, pool_smiles[p_idx], p_idx))
        if (vi + 1) % 500 == 0:
            print(f"  {vi+1}/{len(val_smiles)} val mols; cumulative pairs={len(pairs):,}; "
                  f"el={time.perf_counter()-t0:.1f}s")
    print(f"  total (val, pool) MCES-pairs: {len(pairs):,} in {time.perf_counter()-t0:.1f}s")

    if not pairs:
        print("  no pairs to check — writing pool unchanged.")
        pool.to_csv(args.out, sep="\t", index=False)
        return

    print(f"\nComputing MCES (threshold=2) on {len(pairs):,} pairs with {args.workers} workers ...")
    t0 = time.perf_counter()
    hit_indices = set()
    ctx = get_context("fork")
    with ctx.Pool(args.workers, initializer=_mces_worker_init) as p:
        n_done = 0
        for p_idx, d in p.imap_unordered(_mces_pair, pairs, chunksize=8):
            n_done += 1
            if d <= 2:
                hit_indices.add(p_idx)
            if n_done % 50_000 == 0:
                el = time.perf_counter() - t0
                rate = n_done / max(el, 1)
                eta = (len(pairs) - n_done) / max(rate, 1) / 60
                print(f"  {n_done:,}/{len(pairs):,}  rate={rate:.1f}/s  ETA={eta:.1f}m  hits={len(hit_indices)}")
    print(f"  MCES done in {time.perf_counter()-t0:.1f}s")
    print(f"  MCES ≤ 2 pool molecules to remove: {len(hit_indices):,}")

    keep_mask = np.array([i not in hit_indices for i in range(len(pool))], dtype=bool)
    filtered = pool[keep_mask].reset_index(drop=True)
    print(f"  filtered pool: {len(filtered):,} / {len(pool):,} retained")

    print(f"\nWriting {args.out} ...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.out, sep="\t", index=False)
    print(f"  done; size {args.out.stat().st_size/1e6:.1f} MB")

    # Verify 2D-IK disjointness
    print(f"\n2D-IK sanity:")
    from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
    def ik2d(smi):
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None: return None
            inchi = MolToInchi(m)
            return InchiToInchiKey(inchi)[:14] if inchi else None
        except Exception:
            return None
    with Pool(args.workers) as p:
        filt_iks = p.map(ik2d, filtered["smiles"].tolist(), chunksize=4000)
    filt_ik_set = {k for k in filt_iks if k is not None}
    msg_full = pd.read_csv(args.msg_tsv, sep="\t", usecols=["inchikey", "fold"])
    for fold in ["train", "val", "test"]:
        iks = set(msg_full[msg_full.fold == fold]["inchikey"].dropna())
        overlap = iks & filt_ik_set
        print(f"  {fold:>5s}: overlap = {len(overlap):,} / {len(iks):,}")


if __name__ == "__main__":
    main()
