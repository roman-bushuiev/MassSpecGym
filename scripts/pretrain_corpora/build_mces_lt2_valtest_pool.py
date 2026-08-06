"""Build a 4M pretraining corpus that is exactly MCES >= 2 from MassSpecGym val AND test.

Input : the published MassSpecGym pool `MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv`
Output: `MassSpecGym_molecules_MCES2_disjoint_with_valtest_fold_4M.tsv` — the same file with every
        molecule at MCES < 2 from any val or test molecule removed.

Strict `< 2`, matching MassSpecGym's own convention (NeurIPS 2024 §2.6 excludes "an MCES distance
of less than two" from test; the fold split likewise merges clusters only below 10).

**Why `threshold=2` is exactly right for a `< 2` rule.** `MCES_ILP` constrains the objective to
`<= threshold` and returns `threshold` itself when the problem is infeasible, so a returned `2.0` is
ambiguous between "exactly 2" and "> 2". It does not matter here: neither is `< 2`, so both are
kept. Values strictly below 2 are ILP-optimal and exact. No sentinel ambiguity can affect the rule.

**Why the candidate screen is exact, not a heuristic.** In the ILP objective
`cost = Σ w(unmapped edges) + Σ |w1-w2| (mapped pairs)` every bond weight is >= 1, so `MCES < 2`
admits at most ONE unmapped bond across both graphs. An unmapped atom must have all its bonds
unmapped ("the mapping of the edges has to match the mapping of the nodes"), so with connected
graphs at most one atom can go unmapped — hence the heavy-atom formula L1 distance is <= 1. We
screen at L1 <= 2 for margin. Connectivity is asserted at build time.

Contrast with the superseded `v1.5_candidates/build_mces2_valtest_disjoint_pool.py`, which screened
on the *hydrogen-inclusive* formula. MCES never sees hydrogens (`construct_graph` does not call
`AddHs`), so that screen silently skipped CH2 homologs (d = 1) and CH3<->halogen swaps (d = 2).

Stages:
    prep      compute heavy-atom formulas once, cache the candidate-search tables
    score     one array shard: enumerate + score its slice of queries (resumable per query)
    assemble  merge shard hits, write the corpus, verify lineage
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
csv.field_size_limit(sys.maxsize)

SLOW_PAIR_SECONDS = 30.0    # log any pair whose solve exceeds this

# Screen soundness, general form. Every bond weight in the MCES objective is >= 1, so `MCES <= T`
# admits at most T unmapped bonds. An unmapped atom needs all its bonds unmapped, and detaching k
# atoms from a CONNECTED graph costs at least k bonds (k-1 internal + 1 attachment), so at most T
# atoms go unmapped and the heavy-atom formula L1 distance is <= T.
#   d < 2   -> at most 1 unmapped bond  -> L1 <= 1   (we screen at 2, margin)
#   d <= 2  -> at most 2                -> L1 <= 2
#   d <= 3  -> at most 3                -> L1 <= 3   <- a wider rule NEEDS a wider screen

# The rule and the solver threshold must be chosen together, because MCES_ILP returns the
# threshold value itself when the problem is infeasible:
#   "< 2"  -> threshold 2 is exact. Below-2 values are ILP-optimal; a returned 2.0 is not < 2
#             whether it means "exactly 2" or "> 2", so the rule is unaffected.
#   "<= 2" -> threshold 2 would be WRONG (every sentinel 2.0 gets removed, including far pairs —
#             this is the bug in the superseded builder). Use threshold 3: then values <= 2 are
#             ILP-optimal and exact, and the sentinel moves to 3.0.
RULES = {
    "lt2":  {"threshold": 2, "remove_below": 2.0, "remove_equal": False, "screen_l1": 2},
    "lte2": {"threshold": 3, "remove_below": 2.0, "remove_equal": True,  "screen_l1": 2},
    # lte3 is a CONTROL, not a corpus rule: if a <= 3 scan also returns nothing, the pipeline is
    # broken rather than the pool being clean.
    "lte3": {"threshold": 4, "remove_below": 3.0, "remove_equal": True,  "screen_l1": 3},
}


def read_tsv_cols(path: Path, cols):
    out = {c: [] for c in cols}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in cols if c not in (rd.fieldnames or [])]
        if missing:
            raise KeyError(f"{path} missing column(s) {missing}; has {rd.fieldnames}")
        for row in rd:
            for c in cols:
                out[c].append(row[c])
    return out


def heavy_formula(smi: str):
    """Element counts over HEAVY atoms only — the graph MCES actually operates on."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        c = Counter()
        for a in m.GetAtoms():
            c[a.GetSymbol()] += 1
        return tuple(sorted(c.items()))
    except Exception:
        return None


def _worker_init():
    import pulp
    from myopic_mces.myopic_mces import MCES

    global _MCES_FN, _SOLVER
    _MCES_FN = MCES
    _SOLVER = pulp.listSolvers(onlyAvailable=True)[0]


def _score(args):
    """No timeLimit — matches MassSpecGym's MyopicMCES default. A time limit would let CBC stop
    early and return the `threshold` sentinel, which reads as "far" and silently under-removes."""
    q_smi, p_smi, p_idx, thr = args
    t0 = time.perf_counter()
    try:
        d = _MCES_FN(s1=q_smi, s2=p_smi, ind=0, threshold=thr, always_stronger_bound=True,
                     solver=_SOLVER, solver_options={"msg": 0})[1]
        return (p_idx, float(d), time.perf_counter() - t0, "")
    except Exception as exc:
        return (p_idx, float("nan"), time.perf_counter() - t0, repr(exc)[:200])


def _vec(key, elements):
    v = np.zeros(len(elements), dtype=np.int32)
    if key is not None:
        d = dict(key)
        for i, e in enumerate(elements):
            v[i] = d.get(e, 0)
    return v


def load_queries(msg_tsv: Path):
    msg = read_tsv_cols(msg_tsv, ["smiles", "fold"])
    per_fold = defaultdict(set)
    for s, f in zip(msg["smiles"], msg["fold"]):
        if f in ("val", "test") and s:
            per_fold[f].add(s)
    queries, folds = [], []
    for f in ("val", "test"):
        for s in sorted(per_fold[f]):
            queries.append(s)
            folds.append(f)
    return queries, folds


def stage_prep(args):
    print(f"Loading pool {args.pool} ...", flush=True)
    pool_smiles = read_tsv_cols(args.pool, ["smiles"])["smiles"]
    print(f"  pool: {len(pool_smiles):,}", flush=True)
    queries, folds = load_queries(args.msg_tsv)
    print(f"  queries: {len(queries):,} "
          f"(val {folds.count('val'):,}, test {folds.count('test'):,})", flush=True)

    bad_pool = sum(1 for s in pool_smiles if "." in s)
    bad_q = sum(1 for s in queries if "." in s)
    print(f"  multi-fragment SMILES: pool={bad_pool}, query={bad_q}", flush=True)
    if bad_pool or bad_q:
        raise SystemExit(
            "ABORT: the heavy-atom L1 <= 2 screen is only exact for connected graphs, and a "
            "disconnected zero-degree atom costs nothing in the MCES objective. Disconnected "
            "inputs found — widen the screen or desalt before rebuilding."
        )

    t0 = time.perf_counter()
    with Pool(args.workers) as p:
        pool_keys = p.map(heavy_formula, pool_smiles, chunksize=4000)
        q_keys = p.map(heavy_formula, queries, chunksize=200)
    n_bad = sum(1 for k in pool_keys if k is None)
    print(f"  heavy formulas in {time.perf_counter()-t0:.1f}s; unparseable pool SMILES: {n_bad}",
          flush=True)

    uniq = sorted({k for k in pool_keys if k is not None})
    elements = sorted({e for k in uniq for e, _ in k} | {e for k in q_keys if k for e, _ in k})
    print(f"  unique heavy formulas: {len(uniq):,}; elements: {elements}", flush=True)

    fid = {k: i for i, k in enumerate(uniq)}
    F_uniq = np.stack([_vec(k, elements) for k in uniq])
    F_q = np.stack([_vec(k, elements) for k in q_keys])
    pool_fidx = np.array([fid.get(k, -1) for k in pool_keys], dtype=np.int32)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, F_uniq=F_uniq, F_q=F_q, pool_fidx=pool_fidx,
                        elements=np.array(elements), queries=np.array(queries),
                        folds=np.array(folds), pool_name=np.array(Path(args.pool).name),
                        pool_rows=np.array(len(pool_smiles)))
    print(f"  wrote {args.cache}", flush=True)

    counts = np.bincount(pool_fidx[pool_fidx >= 0], minlength=len(uniq))
    sizes = {}
    for screen in sorted({r["screen_l1"] for r in RULES.values()}):
        tot = 0
        for qi in range(len(queries)):
            near = np.where(np.abs(F_uniq - F_q[qi][None, :]).sum(axis=1) <= screen)[0]
            tot += int(counts[near].sum())
        sizes[screen] = tot
        print(f"\nCANDIDATE PAIRS (heavy L1 <= {screen}): {tot:,}", flush=True)
        print(f"  ~{tot/len(queries):,.0f} pool molecules per query", flush=True)
    tot = sizes[min(sizes)]
    print(f"  brute force for comparison: {len(pool_smiles)*len(queries):,}", flush=True)
    (args.cache.with_suffix(".plan.json")).write_text(json.dumps({
        "pool": str(args.pool), "pool_size": len(pool_smiles), "n_queries": len(queries),
        "n_val": folds.count("val"), "n_test": folds.count("test"),
        "candidate_pairs_by_screen": sizes,
        "bruteforce_pairs": len(pool_smiles) * len(queries),
    }, indent=2))


def stage_score(args):
    z = np.load(args.cache, allow_pickle=False)
    F_uniq, F_q, pool_fidx = z["F_uniq"], z["F_q"], z["pool_fidx"]
    queries = [str(s) for s in z["queries"]]
    folds = [str(f) for f in z["folds"]]
    pool_smiles = read_tsv_cols(args.pool, ["smiles"])["smiles"]
    # The cache indexes the pool by ROW ORDER. Scoring a different file, or one whose row count
    # changed, would silently map every hit to the wrong molecule.
    if "pool_rows" in z and int(z["pool_rows"]) != len(pool_smiles):
        raise SystemExit(f"ABORT: cache built from {int(z['pool_rows']):,} rows, --pool has "
                         f"{len(pool_smiles):,}. Re-run --stage prep.")
    if "pool_name" in z and str(z["pool_name"]) != Path(args.pool).name:
        raise SystemExit(f"ABORT: cache built from {str(z['pool_name'])}, not "
                         f"{Path(args.pool).name}. Re-run --stage prep.")

    by_formula = defaultdict(list)
    for idx, fi in enumerate(pool_fidx):
        if fi >= 0:
            by_formula[int(fi)].append(idx)

    rule = RULES[args.rule]
    thr, below, eq = rule["threshold"], rule["remove_below"], rule["remove_equal"]
    screen = rule["screen_l1"]
    print(f"  rule={args.rule}: remove d < {below}" + (f" or d == {below}" if eq else "")
          + f"; solver threshold={thr}; heavy-atom L1 screen <= {screen}", flush=True)

    def is_hit(d):
        return d < below - 1e-9 or (eq and abs(d - below) <= 1e-9)

    mine = list(range(args.shard, len(queries), args.nshards))
    out_hits = args.out_dir / f"hits_{args.shard:04d}.tsv"
    out_done = args.out_dir / f"done_{args.shard:04d}.txt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_done.exists():
        done = {int(x) for x in out_done.read_text().split()}
        print(f"  resuming: {len(done):,}/{len(mine):,} queries already scored", flush=True)

    n_pairs = n_hits = n_err = 0
    slow = []
    t_start = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(args.workers, initializer=_worker_init) as pool, \
            open(out_hits, "a", newline="") as fh, open(out_done, "a") as fd:
        w = csv.writer(fh, delimiter="\t")
        if out_hits.stat().st_size == 0:
            w.writerow(["query_smiles", "pool_smiles", "pool_idx", "mces", "fold"])
        for n, qi in enumerate(mine, 1):
            if qi in done:
                continue
            near = np.where(np.abs(F_uniq - F_q[qi][None, :]).sum(axis=1) <= screen)[0]
            cand = [i for fi in near for i in by_formula.get(int(fi), ())]
            q_smi = queries[qi]
            tasks = [(q_smi, pool_smiles[i], i, thr) for i in cand]
            n_pairs += len(tasks)
            for p_idx, d, secs, err in pool.imap_unordered(_score, tasks, chunksize=8):
                if err:
                    n_err += 1
                if secs > SLOW_PAIR_SECONDS:
                    slow.append({"query": q_smi, "pool": pool_smiles[p_idx], "seconds": round(secs, 1)})
                if is_hit(d):
                    n_hits += 1
                    w.writerow([q_smi, pool_smiles[p_idx], p_idx, d, folds[qi]])
            fh.flush()
            fd.write(f"{qi}\n")
            fd.flush()
            if n % 10 == 0:
                el = time.perf_counter() - t_start
                print(f"  shard {args.shard}: {n}/{len(mine)} queries, {n_pairs:,} pairs, "
                      f"{n_hits} hits, {el/60:.1f}m elapsed, "
                      f"ETA {el/max(n,1)*(len(mine)-n)/60:.1f}m", flush=True)

    (args.out_dir / f"stats_{args.shard:04d}.json").write_text(json.dumps({
        "shard": args.shard, "nshards": args.nshards, "rule": args.rule,
        "solver_threshold": thr, "queries": len(mine),
        "pairs": n_pairs, "hits": n_hits, "errors": n_err,
        "slow_pairs": slow, "seconds": round(time.perf_counter() - t_start, 1),
    }, indent=2))
    print(f"shard {args.shard} DONE pairs={n_pairs:,} hits={n_hits:,} errors={n_err}", flush=True)


def stage_assemble(args):
    # A shard killed mid-query re-runs that query on resume, so the same (query, pool) hit can be
    # appended twice. Harmless for the drop set, but it would inflate the reported pair counts.
    seen, hits, stats = set(), [], []
    n_dup = 0
    for f in sorted(args.out_dir.glob("hits_*.tsv")):
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                key = (row["query_smiles"], row["pool_idx"])
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                hits.append(row)
    if n_dup:
        print(f"  dropped {n_dup:,} duplicate hit rows from resumed shards", flush=True)
    for f in sorted(args.out_dir.glob("stats_*.json")):
        stats.append(json.loads(f.read_text()))
    n_shards_done = len(stats)
    tot_pairs = sum(s["pairs"] for s in stats)
    tot_err = sum(s["errors"] for s in stats)
    slow = [s for st in stats for s in st["slow_pairs"]]
    print(f"shards complete: {n_shards_done}; pairs scored: {tot_pairs:,}; "
          f"errors: {tot_err}; slow pairs (>{SLOW_PAIR_SECONDS:.0f}s): {len(slow)}", flush=True)
    if n_shards_done != args.nshards:
        raise SystemExit(f"ABORT: {args.nshards - n_shards_done} shard(s) missing — do not "
                         f"assemble a partial result.")
    if tot_err:
        raise SystemExit(f"ABORT: {tot_err} pair(s) raised; the corpus would be under-filtered.")

    drop = {int(h["pool_idx"]) for h in hits}
    pool_smiles = read_tsv_cols(args.pool, ["smiles"])["smiles"]
    bad = [h for h in hits if pool_smiles[int(h["pool_idx"])] != h["pool_smiles"]]
    if bad:
        raise SystemExit(f"ABORT: {len(bad)} hit row(s) have a pool_idx that does not point at the "
                         f"recorded SMILES — index/order mismatch between score and assemble.")
    print(f"  index check: all {len(hits):,} hit rows map to their recorded SMILES", flush=True)
    by_fold = Counter(h["fold"] for h in hits)
    drop_val = {int(h["pool_idx"]) for h in hits if h["fold"] == "val"}
    drop_test = {int(h["pool_idx"]) for h in hits if h["fold"] == "test"}
    dist = Counter(round(float(h["mces"]), 3) for h in hits)

    print(f"\npairs at MCES < 2: {len(hits):,}  ({dict(by_fold)})", flush=True)
    print(f"distinct pool molecules to remove: {len(drop):,} "
          f"(val {len(drop_val):,}, test {len(drop_test):,}, both {len(drop_val & drop_test):,})",
          flush=True)
    print(f"exact distance histogram: {dict(sorted(dist.items()))}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with open(args.pool, newline="") as fin, open(args.out, "w", newline="") as fout:
        header = fin.readline()
        fout.write(header)
        for i, line in enumerate(fin):
            n_in += 1
            if i not in drop:
                fout.write(line)
                n_out += 1
    print(f"\nwrote {args.out}\n  {n_in:,} -> {n_out:,} molecules (removed {n_in-n_out:,})",
          flush=True)
    if n_in - n_out != len(drop):
        raise SystemExit("ABORT: removed-row count does not match the drop set.")

    summary = {
        "pool": str(args.pool), "out": str(args.out),
        "pool_size": n_in, "corpus_size": n_out, "removed": n_in - n_out,
        "pairs_scored": tot_pairs, "pairs_below_2": len(hits),
        "removed_due_to_val": len(drop_val), "removed_due_to_test": len(drop_test),
        "removed_due_to_both": len(drop_val & drop_test),
        "distance_histogram": {str(k): v for k, v in sorted(dist.items())},
        "solver_errors": tot_err, "slow_pairs": slow[:50],
    }
    (args.out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    with open(args.out_dir / "distances_below_2.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["query_smiles", "pool_smiles", "pool_idx", "mces", "fold"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(sorted(hits, key=lambda h: (h["fold"], float(h["mces"]))))
    print(json.dumps(summary, indent=2)[:2000], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "score", "assemble"])
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--msg-tsv", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--rule", choices=sorted(RULES), default="lt2",
                    help="lt2 = remove MCES < 2 (MassSpecGym's stated rule); "
                         "lte2 = remove MCES <= 2 (what the published artifact actually is)")
    ap.add_argument("--workers", type=int, default=128)
    args = ap.parse_args()
    if args.stage == "prep":
        stage_prep(args)
    elif args.stage == "score":
        stage_score(args)
    else:
        stage_assemble(args)


if __name__ == "__main__":
    main()
