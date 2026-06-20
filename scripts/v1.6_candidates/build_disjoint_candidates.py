#!/usr/bin/env python
"""Build MassSpecGym v1.6 retrieval candidates with cross-fold-DISJOINT decoy pools.

v1.5 draws every query's decoys from one shared S4 pool, so ~25-30% of each fold's distinct decoy
structures recur as a train decoy (decoy↔decoy leakage). v1.6 partitions the candidate molecule
universe across the three folds (train/val/test) so the per-fold decoy pools are mutually disjoint,
then re-emits each query's list from its fold's share (cap 512, query at [0], uniform-random trim
like v1.5).

Two criteria (emitted in one run, into two output dirs):
  - inchi : folds disjoint by exact 2D-InChIKey (each molecule lives in exactly one fold).
  - mces  : folds disjoint by MCES>=2 (molecules within MCES<2 are clustered and co-assigned, so
            single-edit near-identicals can't straddle folds either). Strict refinement of inchi.

Pool = the same S4 buckets the v1.5 candidates were drawn from
(experiments/data_builds/MassSpecGym_S4_v3_1B/buckets.parquet), streamed + filtered to the formulas
present among MassSpecGym queries, reservoir-sampled to <= --per-formula-cap molecules per formula
(uniform, bounds memory; cap >> 3*512 so all three folds can still fill). Reuses
build_msg_s4_candidates emit semantics (formula = exact molecular formula; mass = +-10 ppm; cap 512)
and filter_by_mces_disjoint MCES machinery (myopic_mces + Tanimoto pre-filter).

Each MassSpecGym molecule is in exactly one fold (structure-based split), so GTs are assigned to
their own fold; pure-decoy units are distributed round-robin to balance the folds toward the cap.
"Fill where possible": rare formulas/masses whose pool < ~cap*|folds| yield shorter lists.

Run in DreaMS-Mol/.venv-genmol (rdkit, pyarrow, pandas, myopic_mces, pulp) after sourcing load_env.sh.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

CAP = 512                      # MAX_CANDIDATES (query at [0] + up to CAP-1 decoys)
PPM_MASS = 10                  # mass-challenge tolerance, matches build_msg_s4_candidates
FOLDS = ["train", "val", "test"]
CLUSTER_TANI = 0.80            # tani criterion: co-assign (separate across folds) molecules with
                               # Morgan Tanimoto >= this; set from --tani-threshold in main()
MAX_CLUSTER_N = 12000          # skip O(N^2) Tanimoto for huge formula buckets (treat as singletons)


def _morgan(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _formula_tani_pairs(task):
    """task = (gidx_list, smiles_list) for ONE formula (self-contained -> no fork+COW blowup).
    Returns same-formula index pairs (gi_a, gi_b) with Morgan Tanimoto >= CLUSTER_TANI."""
    gi, smis = task
    fps = [_morgan(s) for s in smis]
    out = []
    for a in range(len(gi)):
        if fps[a] is None:
            continue
        vb = [b for b in range(a + 1, len(gi)) if fps[b] is not None]
        if not vb:
            continue
        sims = BulkTanimotoSimilarity(fps[a], [fps[b] for b in vb])
        for b, s in zip(vb, sims):
            if s >= CLUSTER_TANI:
                out.append((gi[a], gi[b]))
    return out


def tani_clusters_global(universe, n_proc):
    """ik2d -> cluster id over the universe: single-linkage union of molecules with Morgan Tanimoto
    >= CLUSTER_TANI within the same formula (parallel, self-contained per-formula tasks; NO MCES).
    Huge formula buckets are left as singletons."""
    from multiprocessing import Pool
    rows = universe.reset_index(drop=True)
    n = len(rows)
    smiles_all = rows["smiles"].tolist()
    formula_all = rows["formula"].tolist()
    groups = {}
    for k in range(n):
        groups.setdefault(formula_all[k], []).append(k)
    tasks, n_huge = [], 0
    for f, gi in groups.items():
        if len(gi) <= 1:
            continue
        if len(gi) > MAX_CLUSTER_N:
            n_huge += 1
            continue
        tasks.append((gi, [smiles_all[k] for k in gi]))
    del smiles_all, formula_all, groups
    print(f"[tani] {len(tasks):,} multi-member formulas, threshold={CLUSTER_TANI} "
          f"({n_huge} huge left as singletons)", flush=True)

    uf = UF(n)
    npairs = 0
    with Pool(n_proc) as pool:
        for sub in pool.imap_unordered(_formula_tani_pairs, tasks, chunksize=16):
            for i, j in sub:
                uf.union(i, j)
                npairs += 1
    print(f"[tani] unioned {npairs:,} Tanimoto>={CLUSTER_TANI} pairs into clusters", flush=True)
    return {rows.at[k, "ik2d"]: uf.find(k) for k in range(n)}


def assign_units(universe, query_table, cluster_map, label):
    """Assign every universe molecule to ONE fold. cluster_map: ik2d->cluster id ('mces'), or None
    ('inchi'). GTs forced to their fold; pure-decoy units round-robin to balance the folds that query
    each formula. Returns dict ik2d -> fold."""
    gt_fold = dict(zip(query_table["ik2d"], query_table["fold"]))
    formula_folds = query_table.groupby("formula")["fold"].agg(lambda s: sorted(set(s))).to_dict()

    iks_all = universe["ik2d"].tolist()
    formula_all = universe["formula"].tolist()
    groups = {}
    for k in range(len(iks_all)):
        groups.setdefault(formula_all[k], []).append(k)

    assign = {}
    t0 = time.perf_counter()
    for fi, (formula, idx) in enumerate(groups.items()):
        present = formula_folds.get(formula)
        if not present:
            continue                                   # no query of this formula -> decoys unused
        iks = [iks_all[k] for k in idx]
        # units: MCES clusters (mces) or singletons (inchi), first-appearance order
        if cluster_map is not None:
            seen, units = {}, []
            for li, ik in enumerate(iks):
                c = cluster_map.get(ik, ik)
                if c not in seen:
                    seen[c] = len(units); units.append([])
                units[seen[c]].append(li)
        else:
            units = [[li] for li in range(len(iks))]

        rr = {f: 0 for f in present}
        free = []
        for u in units:
            gfolds = {gt_fold[iks[li]] for li in u if iks[li] in gt_fold}
            if gfolds:
                f = sorted(gfolds)[0]                   # GTs of different folds can't co-cluster (MCES>=10>2)
                for li in u:
                    assign[iks[li]] = f
                rr[f] = rr.get(f, 0) + len(u)
            else:
                free.append(u)
        for u in free:
            f = min(present, key=lambda ff: rr[ff])
            for li in u:
                assign[iks[li]] = f
            rr[f] += len(u)
        if (fi + 1) % 4000 == 0:
            print(f"[assign:{label}] {fi+1:,}/{len(groups):,} formulas ({time.perf_counter()-t0:.0f}s)", flush=True)
    return assign


def emit_formula(queries, universe, assign, rng):
    uni = universe.assign(fold=universe["ik2d"].map(assign)).dropna(subset=["fold"])
    by = {(f, fo): (g["smiles"].tolist(), g["ik2d"].tolist())
          for (f, fo), g in uni.groupby(["formula", "fold"], sort=False)}
    out, sizes, trivial = {}, [], 0
    for key, q_ik, fold, formula in zip(queries["smiles"], queries["ik2d"], queries["fold"], queries["formula"]):
        g = by.get((formula, fold))
        if g is None:
            out[key] = [key]; sizes.append(1); trivial += 1; continue
        decoys = [s for s, ik in zip(g[0], g[1]) if ik != q_ik]
        if len(decoys) > CAP - 1:
            decoys = rng.sample(decoys, CAP - 1)
        out[key] = [key] + decoys
        sizes.append(len(out[key])); trivial += int(len(out[key]) == 1)
    return out, sizes, trivial


def emit_mass(queries, universe, assign, rng):
    uni = universe.assign(fold=universe["ik2d"].map(assign)).dropna(subset=["fold"])
    per = {}
    for fold, g in uni.groupby("fold"):
        gs = g.sort_values("exact_mass")
        per[fold] = (gs["exact_mass"].to_numpy(), gs["smiles"].to_numpy(), gs["ik2d"].to_numpy())
    out, sizes, trivial = {}, [], 0
    for key, q_ik, fold, q_mass in zip(queries["smiles"], queries["ik2d"], queries["fold"], queries["exact_mass"]):
        p = per.get(fold)
        if p is None or q_mass != q_mass:
            out[key] = [key]; sizes.append(1); trivial += 1; continue
        masses, smi_s, ik_s = p
        tol = q_mass * PPM_MASS * 1e-6
        lo = int(np.searchsorted(masses, q_mass - tol, "left"))
        hi = int(np.searchsorted(masses, q_mass + tol, "right"))
        decoys = [smi_s[i] for i in range(lo, hi) if ik_s[i] != q_ik]
        if len(decoys) > CAP - 1:
            decoys = rng.sample(decoys, CAP - 1)
        out[key] = [key] + decoys
        sizes.append(len(out[key])); trivial += int(len(out[key]) == 1)
    return out, sizes, trivial


def _stats(sizes):
    a = np.asarray(sizes)
    return (f"median={int(np.median(a))} mean={a.mean():.1f} p90={int(np.percentile(a,90))} "
            f"max={int(a.max())} at_cap={int((a>=CAP).sum())}/{len(a)} ({100*(a>=CAP).mean():.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, type=Path)
    ap.add_argument("--mgf", type=Path, help="spectra MGF to copy into each output dir (same spectra as v1.5)")
    ap.add_argument("--buckets", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path, help="parent of v1.6inchi/ and v1.6mces/")
    ap.add_argument("--criteria", nargs="+", default=["inchi", "tani80"])
    ap.add_argument("--per-formula-cap", type=int, default=2000)  # >= 3*511 so all 3 folds can fill to cap
    ap.add_argument("--tani-threshold", type=float, default=0.80, help="Tanimoto clustering threshold for tani* criteria")
    ap.add_argument("--n-workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global CLUSTER_TANI
    CLUSTER_TANI = args.tani_threshold

    print(f"[load] queries from {args.tsv}", flush=True)
    df = pd.read_csv(args.tsv, sep="\t", usecols=["smiles", "inchikey", "formula", "parent_mass", "fold"])
    queries = df.drop_duplicates("smiles").reset_index(drop=True)
    queries["ik2d"] = queries["inchikey"].astype(str).str[:14]
    queries = queries.rename(columns={"parent_mass": "exact_mass"})
    assert queries.groupby("smiles")["fold"].nunique().max() == 1, "a molecule spans >1 fold"
    q_formulas = set(queries["formula"])
    print(f"[load] {len(queries):,} query molecules, {len(q_formulas):,} formulas; "
          f"folds={queries['fold'].value_counts().to_dict()}", flush=True)

    print(f"[load] streaming pool {args.buckets} (per-formula cap={args.per_formula_cap}) ...", flush=True)
    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    dataset = ds.dataset(str(args.buckets), format="parquet")
    scanner = dataset.scanner(columns=["smiles", "inchikey_2d", "formula", "exact_mass"],
                              filter=ds.field("formula").isin(list(q_formulas)))
    res, cnt, seen, n = {}, {}, set(), 0
    cap = args.per_formula_cap
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
        if n % 50_000_000 < len(d["smiles"]):
            print(f"[load] scanned {n:,} ...", flush=True)
    rows = [(smi, ik, fo, em) for fo, lst in res.items() for (smi, ik, em) in lst]
    pool = pd.DataFrame(rows, columns=["smiles", "ik2d", "formula", "exact_mass"])
    print(f"[load] scanned {n:,}; pool reservoir = {len(pool):,} unique mols across {len(res):,} formulas "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # universe = pool ∪ query GTs (some GTs may be absent from the pool; append them)
    have = set(pool["ik2d"])
    gt_rows = queries[~queries["ik2d"].isin(have)][["smiles", "ik2d", "formula", "exact_mass"]]
    universe = pd.concat([pool, gt_rows], ignore_index=True).drop_duplicates("ik2d").reset_index(drop=True)
    print(f"[universe] {len(universe):,} molecules (pool + {len(gt_rows):,} pool-absent GTs)", flush=True)

    qt = queries[["ik2d", "fold", "formula"]]
    cluster_map = None
    if any(c != "inchi" for c in args.criteria):
        print(f"[tani] computing Tanimoto>={CLUSTER_TANI} clusters over universe ...", flush=True)
        tc = time.perf_counter()
        cluster_map = tani_clusters_global(universe, args.n_workers)
        nclust = len(set(cluster_map.values()))
        print(f"[tani] {nclust:,} clusters over {len(cluster_map):,} mols ({time.perf_counter()-tc:.0f}s)", flush=True)

    for criterion in args.criteria:
        print(f"\n========== criterion = {criterion} ==========", flush=True)
        assign = assign_units(universe, qt, None if criterion == "inchi" else cluster_map, criterion)
        out_dir = args.out_root / f"v1.6{criterion}"
        out_dir.mkdir(parents=True, exist_ok=True)
        e_rng = random.Random(args.seed)
        of, sf, tf = emit_formula(queries, universe, assign, e_rng)
        print(f"[emit_formula:{criterion}] trivial={tf:,}  {_stats(sf)}", flush=True)
        (out_dir / "MassSpecGym1.6_retrieval_candidates_formula.json").write_text(json.dumps(of))
        om, sm, tm = emit_mass(queries, universe, assign, e_rng)
        print(f"[emit_mass:{criterion}]    trivial={tm:,}  {_stats(sm)}", flush=True)
        (out_dir / "MassSpecGym1.6_retrieval_candidates_mass.json").write_text(json.dumps(om))
        shutil.copy(args.tsv, out_dir / "MassSpecGym1.6.tsv")
        if args.mgf and args.mgf.exists():
            shutil.copy(args.mgf, out_dir / "MassSpecGym1.6.mgf")
        print(f"[done:{criterion}] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
