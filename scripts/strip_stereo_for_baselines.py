#!/usr/bin/env python
"""Strip stereo from the MSG TSV + final S4-augmented candidate JSONs so the
spectrum-free retrieval baselines can be evaluated without the stereo-shortcut.

Thin CLI wrapper over ``massspecgym.utils.strip_stereo_{tsv,candidates_json}``;
the actual stripping logic lives in the package.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from massspecgym.utils import (
    strip_stereo_candidates_json,
    strip_stereo_mgf,
    strip_stereo_tsv,
)

DATA = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data")

TSV_IN  = DATA / "MassSpecGym1.5.tsv"
TSV_OUT = DATA / "MassSpecGym1.5_nostereo.tsv"
MGF_IN  = DATA / "MassSpecGym1.5.mgf"
MGF_OUT = DATA / "MassSpecGym1.5_nostereo.mgf"
CANDS = [
    (DATA / "MassSpecGym_S4plusPCplusMol_retrieval_candidates_formula.json",
     DATA / "MassSpecGym_S4plusPCplusMol_retrieval_candidates_formula_nostereo.json"),
    (DATA / "MassSpecGym_S4plusPC_retrieval_candidates_mass.json",
     DATA / "MassSpecGym_S4plusPC_retrieval_candidates_mass_nostereo.json"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("WORKERS", 32)))
    p.add_argument("--skip-mgf", action="store_true", help="Skip MGF stripping (large file, slow).")
    args = p.parse_args()

    print(f"\n[TSV] {TSV_IN.name} -> {TSV_OUT.name}")
    t0 = time.perf_counter()
    n = strip_stereo_tsv(TSV_IN, TSV_OUT, n_workers=args.n_workers)
    print(f"  wrote {n:,} rows in {time.perf_counter()-t0:.1f}s")

    if MGF_IN.exists() and not args.skip_mgf:
        print(f"\n[MGF] {MGF_IN.name} -> {MGF_OUT.name}")
        t0 = time.perf_counter()
        n = strip_stereo_mgf(MGF_IN, MGF_OUT)
        print(f"  rewrote {n:,} SMILES headers in {time.perf_counter()-t0:.1f}s")
    elif args.skip_mgf:
        print(f"\n[MGF] skipped via --skip-mgf")

    for in_p, out_p in CANDS:
        print(f"\n[JSON] {in_p.name} -> {out_p.name}")
        t0 = time.perf_counter()
        n_keys, n_collisions = strip_stereo_candidates_json(in_p, out_p, dedup=True, n_workers=args.n_workers)
        print(f"  wrote {n_keys:,} keys (collisions merged: {n_collisions:,}) in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
