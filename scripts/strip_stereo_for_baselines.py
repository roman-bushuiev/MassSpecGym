#!/usr/bin/env python
"""Strip stereochemistry from MSG queries and S4-augmented candidate JSONs.

Motivation
----------
The S4 SMILES tokenizer (38 tokens from ChEMBL31) has no stereo tokens
(``@``, ``@@``, ``/``, ``\\``), so S4-generated candidates are stereo-stripped
by construction. MSG queries (and PubChem-v1.5 candidates) retain stereo —
this asymmetry is a "spurious signal" that any model can exploit (e.g. the
chirality-count baseline scores ~93% hit@1 vs ~0.1% random for 1024 cap).

This script writes stereo-free siblings:

  data/MassSpecGym1.5_nostereo.tsv
  data/MassSpecGym_S4plusPCplusMol_retrieval_candidates_formula_nostereo.json
  data/MassSpecGym_S4plusPC_retrieval_candidates_mass_nostereo.json

Stripping is done via ``Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)``.
Within each candidate list we then dedup (some former stereoisomer pairs
collapse) and preserve query at index 0.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATA = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data")
TSV_IN = DATA / "MassSpecGym1.5.tsv"
TSV_OUT = DATA / "MassSpecGym1.5_nostereo.tsv"
CANDS = [
    (DATA / "MassSpecGym_S4plusPCplusMol_retrieval_candidates_formula.json",
     DATA / "MassSpecGym_S4plusPCplusMol_retrieval_candidates_formula_nostereo.json"),
    (DATA / "MassSpecGym_S4plusPC_retrieval_candidates_mass.json",
     DATA / "MassSpecGym_S4plusPC_retrieval_candidates_mass_nostereo.json"),
]


def strip_stereo(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def _worker_init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _worker_strip(smi):
    return smi, strip_stereo(smi)


def _strip_in_pool(smiles, n_workers):
    items = list(smiles)
    if n_workers <= 1:
        return dict(_worker_strip(s) for s in items)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        return dict(pool.imap_unordered(_worker_strip, items, chunksize=2000))


def process_tsv(n_workers: int) -> dict[str, str | None]:
    print(f"\n[TSV] reading {TSV_IN}")
    df = pd.read_csv(TSV_IN, sep="\t")
    print(f"  {len(df):,} rows; unique SMILES: {df['smiles'].nunique():,}")
    unique_smi = df["smiles"].dropna().unique().tolist()
    t0 = time.perf_counter()
    print(f"  stripping stereo on {len(unique_smi):,} unique SMILES with {n_workers} workers ...")
    mapping = _strip_in_pool(unique_smi, n_workers)
    print(f"  done in {time.perf_counter()-t0:.1f}s; non-null: {sum(1 for v in mapping.values() if v):,}")

    df["smiles"] = df["smiles"].map(mapping)
    n_before = len(df)
    df = df[df["smiles"].notna()].reset_index(drop=True)
    print(f"  rows after dropping unparseable: {len(df):,} (dropped {n_before-len(df)})")
    print(f"  unique stripped SMILES: {df['smiles'].nunique():,}")
    print(f"  per-fold unique: " + " | ".join(f"{f}={df[df['fold']==f]['smiles'].nunique():,}" for f in ["train", "val", "test"]))

    df.to_csv(TSV_OUT, sep="\t", index=False)
    print(f"  wrote {TSV_OUT}")
    return mapping


def process_json(in_path: Path, out_path: Path, n_workers: int) -> None:
    print(f"\n[JSON] {in_path.name}")
    t0 = time.perf_counter()
    with in_path.open() as f:
        d = json.load(f)
    print(f"  loaded {len(d):,} keys in {time.perf_counter()-t0:.1f}s")
    # Collect all SMILES (keys + candidates) for parallel strip.
    smi_set = set(d.keys())
    for v in d.values():
        smi_set.update(v)
    print(f"  unique SMILES across keys + candidates: {len(smi_set):,}")
    print(f"  stripping stereo with {n_workers} workers ...")
    t0 = time.perf_counter()
    mapping = _strip_in_pool(smi_set, n_workers)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # Rebuild JSON. Multiple original keys can collapse to the same stripped key;
    # we union their candidate lists, preserving the first occurrence of each new key.
    out: dict[str, list[str]] = {}
    n_collisions = 0
    n_cand_dedup = 0
    n_cand_total = 0
    for q, cands in d.items():
        q_strip = mapping.get(q)
        if not q_strip:
            continue
        cand_stripped: list[str] = []
        seen = {q_strip}
        for c in cands:
            cs = mapping.get(c)
            if not cs or cs in seen:
                n_cand_dedup += 1
                continue
            cand_stripped.append(cs)
            seen.add(cs)
            n_cand_total += 1
        new_list = [q_strip] + cand_stripped
        if q_strip in out:
            n_collisions += 1
            # Union: append unseen new cands to the existing list, preserving order.
            existing_seen = set(out[q_strip])
            for c in new_list[1:]:
                if c not in existing_seen and c != q_strip:
                    out[q_strip].append(c)
                    existing_seen.add(c)
        else:
            out[q_strip] = new_list

    import numpy as np
    L = [len(v) for v in out.values()]
    print(f"  keys after stripping: {len(out):,} (collisions merged: {n_collisions:,})")
    print(f"  intra-list dedup drops: {n_cand_dedup:,} of {n_cand_total + n_cand_dedup:,}")
    print(f"  list size: median={int(np.median(L))} mean={sum(L)/len(L):.1f} max={max(L)}")
    print(f"  trivial (len=1): {sum(1 for x in L if x==1):,}")
    print(f"  len<8:           {sum(1 for x in L if x<8):,}")

    with out_path.open("w") as f:
        json.dump(out, f)
    print(f"  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("WORKERS", 32)))
    args = p.parse_args()

    process_tsv(args.n_workers)
    for in_p, out_p in CANDS:
        process_json(in_p, out_p, args.n_workers)


if __name__ == "__main__":
    main()
