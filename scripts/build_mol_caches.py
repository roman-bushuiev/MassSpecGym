#!/usr/bin/env python
"""Build SMILES → MorganFP (HDF5) and SMILES → 2D-InChIKey (pkl) caches
used by retrieval baselines to avoid per-batch RDKit fingerprinting/hashing.

Inputs:
  --tsv            MassSpecGym TSV (uses ``smiles`` column)
  --cands          one or more candidate JSON files (keys + values are SMILES)
  --out-fp-cache   HDF5 path for Morgan fingerprint cache
  --out-ik-cache   PKL path for 2D-InChIKey cache
  --fp-size        Morgan FP size (default 4096; matches DeepSetsRetrieval default)
  --radius         Morgan radius (default 2)
  --n-workers      parallel workers (default min(64, CPU))

Reads every unique SMILES across TSV + all candidate JSONs and writes both caches.
Both ``InMemCachedMolTransform`` instances point at these files at training time.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from massspecgym.data.transforms import InMemCachedMolTransform, MolFingerprinter, MolToInChIKey, MolTransform


class TorchMolFingerprinter(MolTransform):
    """MolFingerprinter that returns a 1D bit-packed torch.Tensor (uint8).

    Storage: 4096 bits / 8 = 512 bytes per FP — 8× smaller than naive uint8.
    Decoding is via :func:`np.unpackbits` at retrieval time. Used to build
    the cache file (HDF5) that fits in RAM across DataLoader workers.
    """

    def __init__(self, fp_size: int = 4096, radius: int = 2):
        self._inner = MolFingerprinter(type="morgan", fp_size=fp_size, radius=radius)
        self.fp_size = fp_size

    def from_smiles(self, smi: str):
        import numpy as np
        arr = self._inner.from_smiles(smi).astype("uint8")
        packed = np.packbits(arr)
        return torch.as_tensor(packed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", type=Path, required=True)
    p.add_argument("--cands", type=Path, nargs="+", required=True)
    p.add_argument("--out-fp-cache", type=Path, required=True)
    p.add_argument("--out-ik-cache", type=Path, required=True)
    p.add_argument("--fp-size", type=int, default=4096)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-workers", type=int, default=min(64, os.cpu_count() or 8))
    args = p.parse_args()

    print(f"Reading TSV {args.tsv}")
    tsv = pd.read_csv(args.tsv, sep="\t", usecols=["smiles"])
    smi_set = set(tsv["smiles"].dropna().tolist())
    print(f"  {len(smi_set):,} unique TSV SMILES")

    for c in args.cands:
        print(f"Reading {c}")
        with c.open() as f:
            d = json.load(f)
        smi_set.update(d.keys())
        for v in d.values():
            smi_set.update(v)
        print(f"  running total unique SMILES: {len(smi_set):,}")

    smiles_list = sorted(smi_set)
    print(f"Total unique SMILES to cache: {len(smiles_list):,}")

    # FP cache (HDF5 tensor format)
    print(f"\nBuilding Morgan FP cache (fp_size={args.fp_size}, radius={args.radius})")
    fp_cache = InMemCachedMolTransform(
        cache_pth=args.out_fp_cache,
        mol_transform=TorchMolFingerprinter(fp_size=args.fp_size, radius=args.radius),
        verbose=True,
        tensor_dtype="uint8",
        output_dtype=torch.float32,
    )
    t0 = time.perf_counter()
    fp_cache.build_cache(smiles_list, num_workers=args.n_workers, force=True)
    print(f"  FP cache built in {time.perf_counter()-t0:.1f}s")

    # IK2D cache (pickle dict format)
    print(f"\nBuilding 2D-InChIKey cache")
    ik_cache = InMemCachedMolTransform(
        cache_pth=args.out_ik_cache,
        mol_transform=MolToInChIKey(twod=True),
        verbose=True,
    )
    t0 = time.perf_counter()
    ik_cache.build_cache(smiles_list, num_workers=args.n_workers, force=True)
    print(f"  IK2D cache built in {time.perf_counter()-t0:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
