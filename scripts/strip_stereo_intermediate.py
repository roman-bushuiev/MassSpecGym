#!/usr/bin/env python
"""Strip stereo from the intermediate per-step candidate JSONs (not just the
final S4+PC+Mol formula and S4+PC mass) so we can run the residual-signal
diagnostic on each augmentation step.

Inputs / outputs:
  S4 formula  → ...formula_nostereo_S4only.json
  S4 mass     → ...mass_nostereo_S4only.json
  S4+PC formula → ...formula_nostereo_S4plusPC.json
  (S4+PC mass is already the final mass; covered by mass_nostereo.json)
  (S4+PC+Mol formula is already the final formula; covered by formula_nostereo.json)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

DATA = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data")
PAIRS = [
    (DATA / "MassSpecGym_S4_retrieval_candidates_formula.json",
     DATA / "MassSpecGym_S4_retrieval_candidates_formula_nostereo.json"),
    (DATA / "MassSpecGym_S4_retrieval_candidates_mass.json",
     DATA / "MassSpecGym_S4_retrieval_candidates_mass_nostereo.json"),
    (DATA / "MassSpecGym_S4plusPC_retrieval_candidates_formula.json",
     DATA / "MassSpecGym_S4plusPC_retrieval_candidates_formula_nostereo.json"),
]


def _worker_init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def strip_stereo(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return smi, None
    try:
        return smi, Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except Exception:
        return smi, None


def _strip_in_pool(smiles_set, n_workers):
    if n_workers <= 1:
        return dict(strip_stereo(s) for s in smiles_set)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        return dict(pool.imap_unordered(strip_stereo, list(smiles_set), chunksize=2000))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("WORKERS", 32)))
    args = p.parse_args()

    for in_path, out_path in PAIRS:
        print(f"\n=== {in_path.name} ===")
        if not in_path.exists():
            print(f"  MISSING input; skipping"); continue
        t0 = time.perf_counter()
        with in_path.open() as f:
            d = json.load(f)
        print(f"  loaded {len(d):,} keys in {time.perf_counter()-t0:.1f}s")
        smi_set = set(d.keys())
        for v in d.values():
            smi_set.update(v)
        print(f"  unique SMILES: {len(smi_set):,}")
        t0 = time.perf_counter()
        mapping = _strip_in_pool(smi_set, args.n_workers)
        print(f"  stripped in {time.perf_counter()-t0:.1f}s")

        out: dict[str, list[str]] = {}
        n_collisions = 0
        for q, cands in d.items():
            q_strip = mapping.get(q)
            if not q_strip:
                continue
            cand_stripped: list[str] = []
            seen = {q_strip}
            for c in cands:
                cs = mapping.get(c)
                if not cs or cs in seen:
                    continue
                cand_stripped.append(cs)
                seen.add(cs)
            new_list = [q_strip] + cand_stripped
            if q_strip in out:
                n_collisions += 1
                existing_seen = set(out[q_strip])
                for c in new_list[1:]:
                    if c not in existing_seen and c != q_strip:
                        out[q_strip].append(c)
                        existing_seen.add(c)
            else:
                out[q_strip] = new_list

        import numpy as np
        L = [len(v) for v in out.values()]
        print(f"  keys after: {len(out):,} (collisions merged: {n_collisions:,})")
        print(f"  list size: median={int(np.median(L))} mean={sum(L)/len(L):.1f} trivial(len=1)={sum(1 for x in L if x==1):,}")
        with out_path.open("w") as f:
            json.dump(out, f)
        print(f"  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
