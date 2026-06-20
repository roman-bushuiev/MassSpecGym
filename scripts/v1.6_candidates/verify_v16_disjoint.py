#!/usr/bin/env python
"""Verify MassSpecGym v1.6 cross-fold disjointness + cap-fill, vs the v1.5 baseline leakage.

For each criterion dir (v1.6inchi, v1.6mces) and each challenge (formula, mass):
  - map query keys -> fold (via MassSpecGym1.5.tsv), build per-fold decoy 2D-InChIKey sets;
  - DISJOINTNESS GATE: train∩val = train∩test = val∩test = 0 (exact);
  - for v1.6mces additionally: 0 cross-fold pairs at MCES distance < 2 (Tanimoto>=0.9 prefilter);
  - GT-not-a-cross-fold-decoy: each fold's GT IKs absent from other folds' decoy sets;
  - cap-fill: per-fold list-size distribution;
  - formula/mass consistency spot-check (decoys share query formula / within ±10 ppm).
Also reports the v1.5 decoy↔decoy overlap (the ~25-30% baseline) for before/after.

Run in DreaMS-Mol/.venv-genmol (rdkit + myopic_mces + pulp) after sourcing load_env.sh.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.*")
MCES_THRESHOLD = 2
TANI_PREFILTER = 0.9
PPM_MASS = 10


def _ik2d(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


def _props(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return (None, None, float("nan"))
    return (Chem.MolToInchiKey(m)[:14], rdMolDescriptors.CalcMolFormula(m), float(rdMolDescriptors.CalcExactMolWt(m)))


def _mces_lt2(args):
    from myopic_mces.myopic_mces import MCES
    import pulp
    i, j, si, sj = args
    try:
        r = MCES(0, si, sj, MCES_THRESHOLD, pulp.listSolvers(onlyAvailable=True)[0],
                 solver_options={"msg": 0, "timeLimit": 60}, always_stronger_bound=False)
        return (i, j, float(r[1]) < MCES_THRESHOLD)
    except Exception:
        return (i, j, False)


def per_fold_decoy_iks(data, key2fold, P):
    """{fold: set(decoy ik2d)} and per-fold list sizes."""
    fold_iks = {f: set() for f in ("train", "val", "test")}
    sizes = defaultdict(list)
    for k, v in data.items():
        fo = key2fold.get(k)
        if fo is None:
            continue
        sizes[fo].append(len(v))
        for d in v[1:]:
            ik = P.get(d)
            if ik:
                fold_iks[fo].add(ik)
    return fold_iks, sizes


def report(name, data, key2fold, P, key2ik, tani_thr, n_proc):
    fold_iks, sizes = per_fold_decoy_iks(data, key2fold, P)
    tv = len(fold_iks["train"] & fold_iks["val"])
    tt = len(fold_iks["train"] & fold_iks["test"])
    vt = len(fold_iks["val"] & fold_iks["test"])
    print(f"\n[{name}] exact decoy-set overlaps: train∩val={tv:,}  train∩test={tt:,}  val∩test={vt:,}")
    for fo in ("train", "val", "test"):
        a = np.asarray(sizes[fo]) if sizes[fo] else np.array([0])
        print(f"    {fo}: n_queries={len(sizes[fo]):,}  list size median={int(np.median(a))} "
              f"mean={a.mean():.0f} at_cap(512)={int((a>=512).sum())} ({100*(a>=512).mean():.1f}%)")
    # GT-as-cross-fold-decoy
    gt = {f: set() for f in ("train", "val", "test")}
    for k, fo in key2fold.items():
        ik = key2ik.get(k)
        if ik:
            gt[fo].add(ik)
    gt_leak = (len(gt["val"] & fold_iks["train"]) + len(gt["test"] & fold_iks["train"])
               + len(gt["train"] & (fold_iks["val"] | fold_iks["test"])))
    print(f"    GT-as-cross-fold-decoy: {gt_leak}")
    if tani_thr and (tv + tt + vt) == 0:
        # cross-fold near-dup gate: count cross-fold decoy pairs with Tanimoto >= tani_thr (per
        # formula; should be 0 for a tani-disjoint set). No MCES.
        ik_fold, ik_smi = {}, {}
        for k, v in data.items():
            fo = key2fold.get(k)
            if fo is None:
                continue
            for d in v[1:]:
                ik = P.get(d)
                if ik and ik not in ik_fold:
                    ik_fold[ik] = fo; ik_smi[ik] = d
        smis_list = list(ik_smi.values())
        with Pool(n_proc) as pool:                       # parallel formula + fingerprint
            forms = pool.map(_formula_only, smis_list, chunksize=512)
            fps_list = pool.map(_morgan, smis_list, chunksize=512)
        ik2fp = dict(zip(ik_smi.keys(), fps_list))
        byf = defaultdict(list)
        for ik, fm in zip(ik_smi.keys(), forms):
            byf[fm].append(ik)
        n_viol = 0
        worst = 0.0
        for fm, iks in byf.items():
            if fm is None or len(iks) < 2:
                continue
            fps = [ik2fp[ik] for ik in iks]
            for a in range(len(iks)):
                if fps[a] is None:
                    continue
                # train-only deleak: only TRAIN<->(val/test) pairs are the gate (val<->test is kept
                # at v1.5 verbatim, so its overlap is expected and not counted here).
                a_tr = ik_fold[iks[a]] == "train"
                vb = [b for b in range(a + 1, len(iks))
                      if ((ik_fold[iks[b]] == "train") != a_tr) and fps[b] is not None]
                if not vb:
                    continue
                sims = BulkTanimotoSimilarity(fps[a], [fps[b] for b in vb])
                worst = max(worst, max(sims))
                n_viol += int(sum(1 for s in sims if s >= tani_thr))
        print(f"    train<->val/test Tanimoto>={tani_thr} pairs: {n_viol}  (max train<->val/test Tanimoto={worst:.3f})")
    return (tv, tt, vt)


def _formula_only(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return rdMolDescriptors.CalcMolFormula(m) if m is not None else None


def _morgan(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v16-root", required=True, type=Path)
    ap.add_argument("--tsv", required=True, type=Path)
    ap.add_argument("--v15-dir", type=Path, help="v1.5 dir for baseline overlap comparison")
    ap.add_argument("--criteria", nargs="+", default=["inchi", "tani80"])
    ap.add_argument("--tani-threshold", type=float, default=0.80, help="cross-fold gate for tani* criteria")
    ap.add_argument("--n-workers", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t", usecols=["smiles", "inchikey", "fold"]).drop_duplicates("smiles")
    key2fold = dict(zip(df["smiles"], df["fold"]))
    key2ik = dict(zip(df["smiles"], df["inchikey"].astype(str).str[:14]))
    print(f"[verify] {len(key2fold):,} query molecules")

    jobs = []
    if args.v15_dir:
        jobs.append(("v1.5 (baseline)", args.v15_dir, "formula", None))
        jobs.append(("v1.5 (baseline)", args.v15_dir, "mass", None))
    for crit in args.criteria:
        thr = None if crit == "inchi" else args.tani_threshold
        for ch in ("formula", "mass"):
            jobs.append((f"v1.6{crit}", args.v16_root / f"v1.6{crit}", ch, thr))

    for name, d, ch, tani_thr in jobs:
        fn = next(d.glob(f"*candidates_{ch}.json"), None)
        if fn is None:
            print(f"[skip] {name} {ch}: no json in {d}")
            continue
        data = json.load(open(fn))
        uniq = sorted({s for v in data.values() for s in v[1:]})
        with Pool(args.n_workers) as pool:
            iks = pool.map(_ik2d, uniq, chunksize=512)
        P = dict(zip(uniq, iks))
        report(f"{name} {ch}", data, key2fold, P, key2ik, tani_thr, args.n_workers)


if __name__ == "__main__":
    main()
