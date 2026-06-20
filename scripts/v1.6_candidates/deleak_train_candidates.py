#!/usr/bin/env python
"""Train-only deleaking of MassSpecGym candidates (v1.6).

Keep the v1.5 val/test candidate lists VERBATIM (so the official eval is unchanged / comparable),
and only clean the TRAIN lists: drop any train decoy that overlaps the val/test DECOY universe, then
backfill from the S4 pool to the 512 cap with non-overlapping same-formula (formula challenge) /
mass-window (mass challenge) molecules.

"Overlap" criterion -> two output dirs:
  - inchi  : train decoy shares a 2D-InChIKey with a val/test decoy.
  - tani80 : train decoy has Morgan Tanimoto >= 0.80 to a val/test decoy of the same formula
             (subsumes the exact-InChIKey case).
val/test GROUND TRUTHS are NOT treated as leakage (project decision) — only val/test DECOYS
(list positions [1:]) form the reference.

Pool = S4 buckets (experiments/data_builds/MassSpecGym_S4_v3_1B/buckets.parquet), streamed + filtered
to train-query formulas, reservoir-capped per formula. Reuses the v1.5 emit conventions (formula =
exact molecular formula; mass = +-10 ppm; cap 512; uniform-random trim; query at [0]).

Run in DreaMS-Mol/.venv-genmol after sourcing load_env.sh.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.*")
CAP = 512
PPM_MASS = 10
MAX_TANI_REF = 12000           # cap on val/test ref fps per formula (huge formulas: subsample ref)


def _ik(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


def _ik_formula(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return (None, None)
    return (Chem.MolToInchiKey(m)[:14], rdMolDescriptors.CalcMolFormula(m))


def _morgan(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None


# ── per-formula Tanimoto leak check (self-contained task -> no fork+COW blowup) ──────────────
def _formula_leak_iks(task):
    """task = (cand_iks, cand_smis, ref_smis). Return the subset of cand_iks whose Morgan Tanimoto
    to ANY ref (val/test decoy of the same formula) is >= TANI_THR."""
    cand_iks, cand_smis, ref_smis = task
    ref_fps = [fp for fp in (_morgan(s) for s in ref_smis) if fp is not None]
    if not ref_fps:
        return []
    out = []
    for ik, smi in zip(cand_iks, cand_smis):
        fp = _morgan(smi)
        if fp is None:
            continue
        if max(BulkTanimotoSimilarity(fp, ref_fps)) >= TANI_THR:
            out.append(ik)
    return out


def leaking_iks_tanimoto(cand_by_formula, ref_by_formula, n_proc):
    """cand_by_formula: {formula: [(ik, smi)]}; ref_by_formula: {formula: [smi]} (val/test decoys).
    Returns set of cand iks that are Tanimoto>=TANI_THR to a same-formula ref."""
    tasks = []
    for f, cands in cand_by_formula.items():
        refs = ref_by_formula.get(f)
        if not refs:
            continue                                   # no val/test decoy of this formula -> no leak
        if len(refs) > MAX_TANI_REF:
            refs = refs[:MAX_TANI_REF]
        tasks.append(([ik for ik, _ in cands], [s for _, s in cands], refs))
    print(f"[tani] {len(tasks):,} formulas with val/test refs to leak-check", flush=True)
    leak = set()
    if tasks:
        with Pool(n_proc) as pool:
            for sub in pool.imap_unordered(_formula_leak_iks, tasks, chunksize=8):
                leak.update(sub)
    return leak


def stream_pool(buckets, formulas, cap, seed):
    """Reservoir-sample the S4 pool to <= cap molecules per formula (memory-bounded)."""
    rng = random.Random(seed)
    scanner = ds.dataset(str(buckets), format="parquet").scanner(
        columns=["smiles", "inchikey_2d", "formula", "exact_mass"],
        filter=ds.field("formula").isin(list(formulas)))
    res, cnt, seen, n = {}, {}, set(), 0
    t0 = time.perf_counter()
    for batch in scanner.to_batches():
        d = batch.to_pydict()
        for smi, ik, fo, em in zip(d["smiles"], d["inchikey_2d"], d["formula"], d["exact_mass"]):
            n += 1
            if ik in seen:
                continue
            seen.add(ik)
            c = cnt.get(fo, 0); cnt[fo] = c + 1
            lst = res.setdefault(fo, [])
            if len(lst) < cap:
                lst.append((smi, ik, em))
            else:
                j = rng.randint(0, c)
                if j < cap:
                    lst[j] = (smi, ik, em)
    rows = [(smi, ik, fo, em) for fo, lst in res.items() for (smi, ik, em) in lst]
    pool = pd.DataFrame(rows, columns=["smiles", "ik2d", "formula", "exact_mass"])
    print(f"[load] pool reservoir = {len(pool):,} mols, {len(res):,} formulas, scanned {n:,} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return pool


def deleak_challenge(v15, queries, pool, criterion, challenge, n_proc, seed):
    key2fold = dict(zip(queries["smiles"], queries["fold"]))
    train_keys = [k for k in v15 if key2fold.get(k) == "train"]
    n_train = len(train_keys)

    # ---- val/test DECOY reference (positions [1:] of val/test lists) ----
    vt_decoys = {d for k, v in v15.items() if key2fold.get(k) in ("val", "test") for d in v[1:]}
    vt_list = sorted(vt_decoys)
    with Pool(n_proc) as p:
        vt_props = p.map(_ik_formula, vt_list, chunksize=512)
    VT_IK = set()
    vt_smi_by_formula = defaultdict(list)
    for smi, (ik, fo) in zip(vt_list, vt_props):
        if ik:
            VT_IK.add(ik)
            vt_smi_by_formula[fo].append(smi)
    print(f"[{criterion}/{challenge}] val/test decoy universe: {len(VT_IK):,} ik2d, "
          f"{len(vt_smi_by_formula):,} formulas", flush=True)

    # ---- molecules that need a clean verdict: v1.5 train decoys + pool ----
    train_decoys = sorted({d for k in train_keys for d in v15[k][1:]})
    with Pool(n_proc) as p:
        td_props = p.map(_ik_formula, train_decoys, chunksize=512)
    td_ik = {s: ik for s, (ik, _) in zip(train_decoys, td_props)}
    td_formula = {s: fo for s, (ik, fo) in zip(train_decoys, td_props)}

    # ---- leak set ----
    if criterion == "inchi":
        leak_ik = set(VT_IK)                            # a molecule leaks iff its ik2d is a vt-decoy ik
    else:  # tani80: ik in VT_IK OR Tanimoto>=thr to a same-formula vt decoy
        vt_formulas = set(vt_smi_by_formula)            # only these formulas can have a tani leak
        cand_by_formula = defaultdict(list)
        for s in train_decoys:                          # train decoys
            if td_ik[s] and td_formula[s] in vt_formulas:
                cand_by_formula[td_formula[s]].append((td_ik[s], s))
        for smi, ik, fo in zip(pool["smiles"], pool["ik2d"], pool["formula"]):  # pool backfill cands
            if fo in vt_formulas:
                cand_by_formula[fo].append((ik, smi))
        leak_ik = VT_IK | leaking_iks_tanimoto(cand_by_formula, vt_smi_by_formula, n_proc)
    print(f"[{criterion}/{challenge}] leaking ik2d set size: {len(leak_ik):,}", flush=True)

    # ---- clean backfill pool ----
    pool_clean = pool[~pool["ik2d"].isin(leak_ik)].reset_index(drop=True)
    rng = random.Random(seed)
    out = {k: v15[k] for k in v15 if key2fold.get(k) in ("val", "test")}  # verbatim val/test

    if challenge == "formula":
        bf = defaultdict(list)
        for smi, ik, fo in zip(pool_clean["smiles"], pool_clean["ik2d"], pool_clean["formula"]):
            bf[fo].append((smi, ik))
        for lst in bf.values():
            rng.shuffle(lst)
    else:  # mass: sorted arrays of the clean pool
        ps = pool_clean.sort_values("exact_mass")
        m_arr = ps["exact_mass"].to_numpy()
        m_smi = ps["smiles"].to_numpy(); m_ik = ps["ik2d"].to_numpy()

    sizes = []
    removed_tot = backfilled_tot = 0
    for k in train_keys:
        q_ik = key2ik_global.get(k)                      # GT 2D-IK (never add as a decoy)
        v = v15[k]
        gt = v[0]
        kept, kept_ik = [], set()
        for d in v[1:]:
            ik = td_ik.get(d)
            if ik is None or ik in leak_ik:
                continue
            if ik in kept_ik:
                continue
            kept.append(d); kept_ik.add(ik)
        removed_tot += (len(v) - 1) - len(kept)
        need = (CAP - 1) - len(kept)
        if need > 0:
            fo = q_formula_global[k]
            if challenge == "formula":
                pooled = bf.get(fo, [])
                for smi, ik in pooled:
                    if need <= 0:
                        break
                    if ik in kept_ik or ik == q_ik:
                        continue
                    kept.append(smi); kept_ik.add(ik); need -= 1; backfilled_tot += 1
            else:
                qm = q_mass_global[k]
                if qm == qm:
                    tol = qm * PPM_MASS * 1e-6
                    lo = int(np.searchsorted(m_arr, qm - tol, "left"))
                    hi = int(np.searchsorted(m_arr, qm + tol, "right"))
                    idx = list(range(lo, hi)); rng.shuffle(idx)
                    for i in idx:
                        if need <= 0:
                            break
                        ik = m_ik[i]
                        if ik in kept_ik or ik == q_ik:
                            continue
                        kept.append(m_smi[i]); kept_ik.add(ik); need -= 1; backfilled_tot += 1
        out[k] = [gt] + kept[:CAP - 1]
        sizes.append(len(out[k]))
    a = np.asarray(sizes)
    print(f"[{criterion}/{challenge}] train: removed {removed_tot:,} leaking decoys, backfilled "
          f"{backfilled_tot:,}; list size median={int(np.median(a))} mean={a.mean():.0f} "
          f"at_cap={int((a>=CAP).sum())}/{n_train} ({100*(a>=CAP).mean():.1f}%)", flush=True)
    return out


# globals for per-query GT/formula/mass lookup (set in main; small dicts, fork-safe)
key2ik_global = {}
q_formula_global = {}
q_mass_global = {}
TANI_THR = 0.80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v15-dir", required=True, type=Path)
    ap.add_argument("--tsv", required=True, type=Path)
    ap.add_argument("--mgf", type=Path)
    ap.add_argument("--buckets", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--criteria", nargs="+", default=["inchi", "tani80"])
    ap.add_argument("--tani-threshold", type=float, default=0.80)
    ap.add_argument("--per-formula-cap", type=int, default=4000)
    ap.add_argument("--n-workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global TANI_THR, key2ik_global, q_formula_global, q_mass_global
    TANI_THR = args.tani_threshold

    df = pd.read_csv(args.tsv, sep="\t", usecols=["smiles", "inchikey", "formula", "parent_mass", "fold"])
    queries = df.drop_duplicates("smiles").reset_index(drop=True)
    queries["ik2d"] = queries["inchikey"].astype(str).str[:14]
    key2ik_global = dict(zip(queries["smiles"], queries["ik2d"]))
    q_formula_global = dict(zip(queries["smiles"], queries["formula"]))
    q_mass_global = dict(zip(queries["smiles"], pd.to_numeric(queries["parent_mass"], errors="coerce")))
    train_formulas = set(queries.loc[queries["fold"] == "train", "formula"])
    print(f"[load] {len(queries):,} query mols; {len(train_formulas):,} train formulas; "
          f"folds={queries['fold'].value_counts().to_dict()}", flush=True)

    pool = stream_pool(args.buckets, train_formulas, args.per_formula_cap, args.seed)

    for criterion in args.criteria:
        out_dir = args.out_root / f"v1.6{criterion}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for challenge in ("formula", "mass"):
            v15_path = next((args.v15_dir).glob(f"*candidates_{challenge}.json"))
            v15 = json.load(open(v15_path))
            out = deleak_challenge(v15, queries, pool, criterion, challenge, args.n_workers, args.seed)
            (out_dir / f"MassSpecGym1.6_retrieval_candidates_{challenge}.json").write_text(json.dumps(out))
        shutil.copy(args.tsv, out_dir / "MassSpecGym1.6.tsv")
        if args.mgf and args.mgf.exists():
            shutil.copy(args.mgf, out_dir / "MassSpecGym1.6.mgf")
        print(f"[done:{criterion}] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
