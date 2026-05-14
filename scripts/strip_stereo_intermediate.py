#!/usr/bin/env python
"""Strip stereo from the intermediate per-step S4 candidate JSONs so the
per-step residual-signal diagnostic can be run on each augmentation step.

Thin CLI wrapper over ``massspecgym.utils.strip_stereo_candidates_json``.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from massspecgym.utils import strip_stereo_candidates_json

DATA = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data")

PAIRS = [
    (DATA / "MassSpecGym_S4_retrieval_candidates_formula.json",
     DATA / "MassSpecGym_S4_retrieval_candidates_formula_nostereo.json"),
    (DATA / "MassSpecGym_S4_retrieval_candidates_mass.json",
     DATA / "MassSpecGym_S4_retrieval_candidates_mass_nostereo.json"),
    (DATA / "MassSpecGym_S4plusPC_retrieval_candidates_formula.json",
     DATA / "MassSpecGym_S4plusPC_retrieval_candidates_formula_nostereo.json"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("WORKERS", 32)))
    args = p.parse_args()
    for in_p, out_p in PAIRS:
        if not in_p.exists():
            print(f"missing: {in_p}; skipping")
            continue
        print(f"\n[JSON] {in_p.name} -> {out_p.name}")
        t0 = time.perf_counter()
        n_keys, n_coll = strip_stereo_candidates_json(in_p, out_p, dedup=True, n_workers=args.n_workers)
        print(f"  wrote {n_keys:,} keys (collisions merged: {n_coll:,}) in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
