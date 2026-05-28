"""Prepare S4 pretrain corpus zips from the MCES2-valtest-disjoint 4M pool
plus MSG-train SMILES (so the tokenizer covers both sources and no OOV
drops occur at finetune time).

Inputs:
  data/pretrain_corpora/MassSpecGym_molecules_MCES2_disjoint_with_valtest_4M.tsv
  data/v1.5/MassSpecGym1.5.tsv  (for train-fold SMILES)

Outputs (placed in --out-dir, conventionally
  experiments/data_builds/MassSpecGym_S4_v2/):
    chembl_std_train.zip   <- keeps the s4dd-pipeline filename convention
    chembl_std_valid.zip

The "chembl_std_" prefix is kept so the existing
DreaMS-Mol/scripts/data_processing/build_s4_candidates.py pretrain stage
reads them without modification. The contents are MCES2-disjoint MSG-4M
mixed with MSG-train, NOT ChEMBL.
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev")
sys.path.insert(0, str(WS / "DreaMS-Mol"))


def _std_one(smi: str):
    from dreams_mol.data.mols import standardize_smiles
    try:
        return standardize_smiles(smi, standardize_tautomers=False, neutralize=True,
                                  strip_stereo=True)
    except Exception:
        return None


def _write_smiles_zip(path: Path, smiles_iter):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(smiles_iter)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.stem + ".txt", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path,
                    default=WS / "MassSpecGym/data/pretrain_corpora"
                              "/MassSpecGym_molecules_MCES2_disjoint_with_valtest_4M.tsv")
    ap.add_argument("--msg-tsv", type=Path,
                    default=WS / "MassSpecGym/data/v1.5/MassSpecGym1.5.tsv")
    ap.add_argument("--out-dir", type=Path,
                    default=WS / "experiments/data_builds/MassSpecGym_S4_v2")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading pool {args.pool} ...")
    pool = pd.read_csv(args.pool, sep="\t", usecols=["smiles"])
    print(f"  pool rows: {len(pool):,}")

    print(f"Loading MSG TSV {args.msg_tsv} ...")
    msg = pd.read_csv(args.msg_tsv, sep="\t", usecols=["smiles", "fold"])
    msg_train_smis = msg[msg.fold == "train"]["smiles"].dropna().unique().tolist()
    print(f"  MSG-train unique mols: {len(msg_train_smis):,}")

    all_smis = pool["smiles"].dropna().tolist() + msg_train_smis
    print(f"  combined raw: {len(all_smis):,}")

    print(f"Standardising (parallel, {args.workers} workers) ...")
    t0 = time.perf_counter()
    with Pool(args.workers) as p:
        std = p.map(_std_one, all_smis, chunksize=4000)
    n_fail = sum(1 for s in std if s is None)
    print(f"  done in {time.perf_counter()-t0:.1f}s; standardise failures: {n_fail:,}")

    seen: set[str] = set()
    clean: list[str] = []
    for s in std:
        if not s or s in seen:
            continue
        seen.add(s)
        clean.append(s)
    print(f"  after dedup: {len(clean):,}")

    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(clean))
    rng.shuffle(idx)
    n_val = max(1, int(len(clean) * args.val_frac))
    val_smis = [clean[i] for i in idx[:n_val]]
    train_smis = [clean[i] for i in idx[n_val:]]
    print(f"  split: train={len(train_smis):,}  val={len(val_smis):,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Filenames match what DreaMS-Mol/scripts/data_processing/build_s4_candidates.py
    # _stage_pretrain reads (chembl_std_train.zip / chembl_std_valid.zip),
    # so we can reuse the existing pretrain stage without code changes.
    train_zip = args.out_dir / "chembl_std_train.zip"
    val_zip = args.out_dir / "chembl_std_valid.zip"
    _write_smiles_zip(train_zip, train_smis)
    _write_smiles_zip(val_zip, val_smis)
    print(f"  wrote {train_zip} ({train_zip.stat().st_size/1e6:.1f} MB)")
    print(f"  wrote {val_zip} ({val_zip.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
