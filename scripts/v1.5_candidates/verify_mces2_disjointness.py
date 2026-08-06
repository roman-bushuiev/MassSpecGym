"""Verify the MCES2-disjointness level of a pretraining molecule pool against an MSG fold.

Audits `build_mces2_valtest_disjoint_pool.py` and the published MassSpecGym pool
`MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv`.

The two questions this answers:

1. **Sign.** MassSpecGym (NeurIPS 2024, Sec. 2.6) excludes molecules with "an MCES
   distance of less than two" from the test fold, i.e. it *retains* d == 2. Our val
   filter removes d <= 2, i.e. retains d > 2. Running this script with
   `--pool <published> --fold test` should therefore surface hits at exactly d == 2
   and none below; with `--pool <ours> --fold val` it should surface none at all.

2. **Completeness of the candidate-pair pre-filter.** MCES distance is computed on the
   *heavy-atom* graph (`myopic_mces.graph.construct_graph` calls `MolFromSmiles`
   without `AddHs`), and every bond weight is >= 1, so d <= 2 implies at most two
   unmapped bonds in total and hence a heavy-atom composition L1 distance <= 2 for
   connected molecules. The builder pre-filtered on the *hydrogen-inclusive* formula
   (`AddHs`) with the same delta of 2, which is a strict subset: a CH2 homolog has
   heavy-L1 = 1 but with-H L1 = 3 and is never even scored. `--prefilter heavy`
   uses the sound bound; `--prefilter withH` reproduces the builder.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, get_context
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

EPS = 1e-6
csv.field_size_limit(sys.maxsize)


def read_tsv_cols(path: Path, cols):
    """Stream a TSV and return {col: [values]} — avoids a pandas dependency on CPU-only nodes."""
    out = {c: [] for c in cols}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in cols if c not in (rd.fieldnames or [])]
        if missing:
            raise KeyError(f"{path} is missing column(s) {missing}; has {rd.fieldnames}")
        for row in rd:
            for c in cols:
                out[c].append(row[c])
    return out


def _counts(smi: str, with_h: bool):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        if with_h:
            m = Chem.AddHs(m)
        c = Counter()
        for a in m.GetAtoms():
            c[a.GetSymbol()] += 1
        return tuple(sorted(c.items()))
    except Exception:
        return None


def formula_heavy(smi: str):
    return _counts(smi, with_h=False)


def formula_withh(smi: str):
    return _counts(smi, with_h=True)


def _mces_worker_init():
    """Byte-identical to `massspecgym.utils.MyopicMCES(threshold=2, always_stronger_bound=True,
    solver_options={"msg": 0, "timeLimit": 10})` as used by the builder, but calling
    `myopic_mces` directly so the audit runs torch-free on a CPU-only node."""
    import pulp
    from myopic_mces.myopic_mces import MCES

    global _MCES_FN, _SOLVER
    _MCES_FN = MCES
    _SOLVER = pulp.listSolvers(onlyAvailable=True)[0]


def _mces_pair(args):
    q_smi, p_smi, q_idx, p_idx = args
    try:
        d = _MCES_FN(
            s1=q_smi, s2=p_smi, ind=0, threshold=2, always_stronger_bound=True,
            solver=_SOLVER, solver_options={"msg": 0, "timeLimit": 10},
        )[1]
        return (q_idx, p_idx, float(d))
    except Exception:
        return (q_idx, p_idx, float("inf"))


def _rescore_worker_init():
    import pulp
    from myopic_mces.myopic_mces import MCES

    global _MCES_FN, _SOLVER
    _MCES_FN = MCES
    _SOLVER = pulp.listSolvers(onlyAvailable=True)[0]


def _rescore_pair(args):
    """Re-measure a pair at a HIGHER threshold to disambiguate the builder's `d <= 2` verdict.

    `MCES_ILP` constrains the objective to <= threshold and returns `threshold` itself when the
    problem is infeasible (true distance above it) or the solver stops early. At threshold=2 a
    returned 2.0 therefore conflates "exactly 2" with "> 2, unproven". Re-running at threshold T
    moves that sentinel to T, so any value < T is the exact distance.
    """
    q_smi, p_smi, thr = args
    try:
        d = _MCES_FN(
            s1=q_smi, s2=p_smi, ind=0, threshold=thr, always_stronger_bound=True,
            solver=_SOLVER, solver_options={"msg": 0, "timeLimit": 60},
        )[1]
        return float(d)
    except Exception:
        return float("nan")


def rescore_hits(path: Path, thr: int, workers: int, out: Path):
    rows = list(csv.DictReader(open(path, newline=""), delimiter="\t"))
    print(f"Re-scoring {len(rows):,} pairs from {path.name} at threshold={thr} ...", flush=True)
    args = [(r["query_smiles"], r["pool_smiles"], thr) for r in rows]
    t0 = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_rescore_worker_init) as p:
        ds = []
        for i, d in enumerate(p.imap(_rescore_pair, args, chunksize=4), 1):
            ds.append(d)
            if i % 200 == 0:
                el = time.perf_counter() - t0
                print(f"  {i:,}/{len(rows):,}  {i/max(el,1):.1f}/s  el={el/60:.1f}m", flush=True)
    hist = Counter(round(d, 3) for d in ds)
    n_exact2 = sum(1 for d in ds if abs(d - 2) <= EPS)
    n_lt2 = sum(1 for d in ds if d < 2 - EPS)
    n_gt2 = sum(1 for d in ds if d > 2 + EPS)
    res = {
        "source": str(path), "rescore_threshold": thr, "n_pairs": len(rows),
        "n_true_lt2": n_lt2, "n_true_eq2": n_exact2, "n_true_gt2": n_gt2,
        "n_at_sentinel": int(hist.get(float(thr), 0)),
        "histogram": {str(k): v for k, v in sorted(hist.items())},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    fields = ["query_smiles", "pool_smiles", "mces_thr2", "mces_rescored"]
    with open(out.with_suffix(".tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r, d in zip(rows, ds):
            w.writerow({"query_smiles": r["query_smiles"], "pool_smiles": r["pool_smiles"],
                        "mces_thr2": r["mces"], "mces_rescored": d})
    print(json.dumps(res, indent=2), flush=True)
    return res


def _vec(key, elements):
    v = np.zeros(len(elements), dtype=np.int32)
    if key is None:
        return v
    d = dict(key)
    for i, e in enumerate(elements):
        v[i] = d.get(e, 0)
    return v


def build_pairs(query_smiles, pool_smiles, prefilter, max_delta, workers):
    """Enumerate (query, pool) pairs whose composition distance is within `max_delta`."""
    fn = formula_withh if prefilter == "withH" else formula_heavy
    print(f"\nComputing {prefilter} formulas ({workers} workers) ...", flush=True)
    t0 = time.perf_counter()
    with Pool(workers) as p:
        pool_keys = p.map(fn, pool_smiles, chunksize=4000)
        q_keys = p.map(fn, query_smiles, chunksize=200)
    print(f"  formulas done in {time.perf_counter()-t0:.1f}s", flush=True)

    unique_formula = list({k for k in pool_keys if k is not None})
    elem_set = set()
    for k in unique_formula:
        elem_set.update(e for e, _ in k)
    for k in q_keys:
        if k is not None:
            elem_set.update(e for e, _ in k)
    elements = sorted(elem_set)
    print(f"  unique pool formulas: {len(unique_formula):,}; elements: {elements}", flush=True)

    F_pool = np.stack([_vec(k, elements) for k in unique_formula])
    F_q = np.stack([_vec(k, elements) for k in q_keys])

    pool_by_formula = defaultdict(list)
    for idx, k in enumerate(pool_keys):
        if k is not None:
            pool_by_formula[k].append(idx)

    print(f"\nMatching queries -> near-formula pool indices (max delta={max_delta}) ...", flush=True)
    t0 = time.perf_counter()
    pairs = []
    for qi in range(len(query_smiles)):
        v = F_q[qi]
        if v.sum() == 0:
            continue
        near = np.where(np.abs(F_pool - v[None, :]).sum(axis=1) <= max_delta)[0]
        q_smi = query_smiles[qi]
        for fi in near:
            for p_idx in pool_by_formula[unique_formula[fi]]:
                pairs.append((q_smi, pool_smiles[p_idx], qi, p_idx))
        if (qi + 1) % 200 == 0:
            print(f"  {qi+1}/{len(query_smiles)} queries; pairs={len(pairs):,}; "
                  f"el={time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  total pairs: {len(pairs):,} in {time.perf_counter()-t0:.1f}s", flush=True)
    return pairs, pool_keys, q_keys


def score_pairs(pairs, workers, report_every=50_000):
    print(f"\nComputing MCES (threshold=2) on {len(pairs):,} pairs, {workers} workers ...", flush=True)
    t0 = time.perf_counter()
    hits = []
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_mces_worker_init) as p:
        n_done = 0
        for q_idx, p_idx, d in p.imap_unordered(_mces_pair, pairs, chunksize=8):
            n_done += 1
            if d <= 2 + EPS:
                hits.append((q_idx, p_idx, d))
            if n_done % report_every == 0:
                el = time.perf_counter() - t0
                rate = n_done / max(el, 1)
                print(f"  {n_done:,}/{len(pairs):,}  rate={rate:.1f}/s  "
                      f"ETA={(len(pairs)-n_done)/max(rate,1)/60:.1f}m  hits={len(hits)}", flush=True)
    print(f"  MCES done in {time.perf_counter()-t0:.1f}s; hits(d<=2)={len(hits):,}", flush=True)
    return hits


def load_pool(path: Path, removed: Path | None):
    smiles = read_tsv_cols(path, ["smiles"])["smiles"]
    if removed is not None:
        drop = set(read_tsv_cols(removed, ["smiles"])["smiles"])
        smiles = [s for s in smiles if s not in drop]
        print(f"  reconstructed cleaned pool by dropping {len(drop):,} SMILES")
    return smiles


# (smiles_a, smiles_b, expected MCES, expected with-H formula L1) — hand-derived from the
# ILP objective (bond weights: single 1.0, aromatic 1.5, double 2.0) on heavy-atom graphs.
SELFTEST = [
    ("Cc1ccccc1", "c1ccccc1", 1.0, 3),    # toluene / benzene   -> builder pre-filter MISSES (L1 3 > 2)
    ("Cc1ccccc1", "Clc1ccccc1", 2.0, 5),  # toluene / chlorobenzene -> MISSES, and sits exactly on d==2
    ("Cc1ccccc1C", "Cc1cccc(C)c1", 2.0, 0),  # o- / m-xylene -> visible (same formula), d==2
    ("CCc1ccccc1", "Cc1ccccc1", 1.0, 3),  # ethyl / methyl homolog -> MISSES (L1 3 > 2)
    ("c1ccccc1", "c1ccccc1", 0.0, 0),     # identical
]


def run_selftest(workers):
    print("=== self-test: MCES semantics + pre-filter visibility ===", flush=True)
    pairs = [(a, b, i, i) for i, (a, b, _, _) in enumerate(SELFTEST)]
    ctx = get_context("fork")
    with ctx.Pool(min(workers, len(pairs)), initializer=_mces_worker_init) as p:
        got = {q: d for q, _, d in p.map(_mces_pair, pairs)}
    ok = True
    for i, (a, b, exp_d, exp_l1) in enumerate(SELFTEST):
        ka, kb = dict(formula_withh(a)), dict(formula_withh(b))
        l1 = sum(abs(ka.get(e, 0) - kb.get(e, 0)) for e in set(ka) | set(kb))
        d = got[i]
        good = abs(d - exp_d) <= EPS and l1 == exp_l1
        ok &= good
        print(f"  {'OK ' if good else 'BAD'}  {a:<14s} {b:<14s}  MCES={d:<5.2f}(exp {exp_d})  "
              f"withH_L1={l1}(exp {exp_l1})  visible_to_builder_prefilter={l1 <= 2}", flush=True)
    print(f"=== self-test {'PASSED' if ok else 'FAILED'} ===\n", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run the MCES semantics self-test and exit")
    ap.add_argument("--rescore-hits", type=Path, default=None,
                    help="a *.hits.tsv from a previous run; re-measure each pair at a higher "
                         "threshold to separate a true distance of 2 from the threshold sentinel")
    ap.add_argument("--rescore-threshold", type=int, default=3)
    ap.add_argument("--pool", type=Path, help="pool TSV with a `smiles` column")
    ap.add_argument("--drop-smiles", type=Path, default=None,
                    help="optional TSV of SMILES to drop from --pool (reconstructs the cleaned pool)")
    ap.add_argument("--msg-tsv", type=Path)
    ap.add_argument("--fold", choices=["train", "val", "test"])
    ap.add_argument("--queries", type=Path, default=None,
                    help="optional TSV of query SMILES instead of an MSG fold (e.g. removed rows)")
    ap.add_argument("--query-source", choices=["fold", "file"], default="fold")
    ap.add_argument("--prefilter", choices=["withH", "heavy"], default="withH")
    ap.add_argument("--max-delta", type=int, default=2)
    ap.add_argument("--n-queries", type=int, default=0, help="0 = all; else random sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="enumerate pairs then stop")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if run_selftest(args.workers) else 1)
    if args.rescore_hits is not None:
        if args.out is None:
            ap.error("--out is required with --rescore-hits")
        rescore_hits(args.rescore_hits, args.rescore_threshold, args.workers, args.out)
        raise SystemExit(0)
    for req in ("pool", "msg_tsv", "fold", "out"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required unless --selftest")

    print(f"Loading pool {args.pool} ...", flush=True)
    pool_smiles = load_pool(args.pool, args.drop_smiles)
    print(f"  pool: {len(pool_smiles):,} molecules", flush=True)

    if args.query_source == "fold":
        msg = read_tsv_cols(args.msg_tsv, ["smiles", "fold"])
        query_smiles = sorted({s for s, f in zip(msg["smiles"], msg["fold"]) if f == args.fold and s})
        query_label = f"MSG-{args.fold}"
    else:
        query_smiles = sorted({s for s in read_tsv_cols(args.queries, ["smiles"])["smiles"] if s})
        query_label = str(args.queries.name)
    print(f"  queries ({query_label}): {len(query_smiles):,}", flush=True)

    n_multifrag_pool = sum(1 for s in pool_smiles if "." in s)
    n_multifrag_q = sum(1 for s in query_smiles if "." in s)
    print(f"  multi-fragment SMILES: pool={n_multifrag_pool:,}, query={n_multifrag_q:,} "
          f"(the heavy-L1<=2 bound assumes connected graphs)", flush=True)

    if args.n_queries and args.n_queries < len(query_smiles):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(query_smiles), size=args.n_queries, replace=False)
        query_smiles = [query_smiles[i] for i in sorted(idx)]
        print(f"  sampled {len(query_smiles):,} queries (seed={args.seed})", flush=True)

    pairs, pool_keys, q_keys = build_pairs(
        query_smiles, pool_smiles, args.prefilter, args.max_delta, args.workers
    )

    result = {
        "pool": str(args.pool),
        "drop_smiles": str(args.drop_smiles) if args.drop_smiles else None,
        "pool_size": len(pool_smiles),
        "query_label": query_label,
        "n_queries": len(query_smiles),
        "prefilter": args.prefilter,
        "max_delta": args.max_delta,
        "seed": args.seed,
        "n_pairs": len(pairs),
        "multifrag_pool": n_multifrag_pool,
        "multifrag_query": n_multifrag_q,
    }
    if args.dry_run:
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    hits = score_pairs(pairs, args.workers)

    # Classify each hit: exact distance bucket, and whether the builder's withH
    # pre-filter would have enumerated this pair at all.
    withh_pool = {}
    withh_q = {}
    if hits:
        uniq_p = sorted({p for _, p, _ in hits})
        uniq_q = sorted({q for q, _, _ in hits})
        with Pool(args.workers) as p:
            wp = p.map(formula_withh, [pool_smiles[i] for i in uniq_p], chunksize=64)
            wq = p.map(formula_withh, [query_smiles[i] for i in uniq_q], chunksize=64)
        withh_pool = dict(zip(uniq_p, wp))
        withh_q = dict(zip(uniq_q, wq))

    def withh_l1(qi, pi):
        a, b = withh_q.get(qi), withh_pool.get(pi)
        if a is None or b is None:
            return None
        da, db = dict(a), dict(b)
        return sum(abs(da.get(e, 0) - db.get(e, 0)) for e in set(da) | set(db))

    dist_hist = Counter()
    rows = []
    n_missed_by_builder = 0
    hit_pool_idx = set()
    for qi, pi, d in hits:
        dist_hist[round(d, 3)] += 1
        l1 = withh_l1(qi, pi)
        missed = l1 is not None and l1 > args.max_delta
        n_missed_by_builder += int(missed)
        hit_pool_idx.add(pi)
        rows.append({
            "query_smiles": query_smiles[qi],
            "pool_smiles": pool_smiles[pi],
            "mces": d,
            "withH_l1": l1,
            "invisible_to_withH_prefilter": missed,
        })

    result.update({
        "n_hits_pairs": len(hits),
        "n_hit_pool_molecules": len(hit_pool_idx),
        "n_hits_lt2": sum(1 for _, _, d in hits if d < 2 - EPS),
        "n_hits_eq2": sum(1 for _, _, d in hits if abs(d - 2) <= EPS),
        "distance_histogram": {str(k): v for k, v in sorted(dist_hist.items())},
        "n_hit_pairs_invisible_to_withH_prefilter": n_missed_by_builder,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    fields = ["query_smiles", "pool_smiles", "mces", "withH_l1", "invisible_to_withH_prefilter"]
    with open(args.out.with_suffix(".hits.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["mces"], r["query_smiles"])))
    print(json.dumps(result, indent=2), flush=True)
    print(f"\nwrote {args.out} and {args.out.with_suffix('.hits.tsv')}", flush=True)


if __name__ == "__main__":
    main()
