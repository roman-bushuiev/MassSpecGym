"""Build a MassSpecGym MGF file directly from a TSV.

Same logic as the conversion step in
``MassSpecGym/scripts/fixes/rdkit_canon_massspecgym.py``: iterate TSV rows,
build a ``matchms.Spectrum`` per row using all non-(mzs/intensities) columns
as metadata, and ``save_as_mgf``. Guarantees that MGF content is a perfect
re-encoding of the TSV.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.exporting import save_as_mgf
from tqdm import tqdm

import massspecgym.utils as utils


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", type=Path, required=True)
    p.add_argument("--mgf", type=Path, required=True)
    args = p.parse_args()

    print(f"Loading TSV {args.tsv}")
    df = pd.read_csv(args.tsv, sep="\t")
    print(f"  {len(df):,} rows; columns: {list(df.columns)}")

    spectra = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building spectra"):
        metadata = {
            k: v for k, v in row.items()
            if k not in ("mzs", "intensities") and v is not np.nan
        }
        spec = Spectrum(
            mz=utils.parse_spec_array(row["mzs"]),
            intensities=utils.parse_spec_array(row["intensities"]),
            metadata=metadata,
        )
        spectra.append(spec)

    if args.mgf.exists():
        print(f"Removing existing {args.mgf}")
        os.remove(args.mgf)
    args.mgf.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.mgf} ({len(spectra):,} spectra)")
    save_as_mgf(spectra, str(args.mgf))
    print("Done.")


if __name__ == "__main__":
    main()
