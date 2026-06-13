#!/usr/bin/env python
"""Build PubChem-by-formula and PubChem-by-mass retrieval candidates for an
arbitrary query set — the reusable replacement for the one-off MassSpecGym
``notebooks/retrieval_candidates_construction/3_generate_retrieval_candidates.ipynb``.

The notebook scanned the full 118M PubChem pool *per query* (O(queries x pool)),
which is infeasible at 300k queries. This script uses the same *indexed*
bucketing as ``build_msg_s4_candidates.py`` emit stages: group the pool by
formula once, mass-sort it once, then look each query up in O(bucket).

It is also restricted to the queries that actually need a PubChem fallback:
PubChem augments only queries whose S4 candidate list has < ``threshold``
entries (see ``augment_s4_with_pubchem.py``). Passing the S4 JSONs via
``--restrict-{formula,mass}-json`` filters the work — and the 30 GB PubChem
scan — down to that short-query tail. Omit them to generate for all queries.

Two stages:
  index — stream the PubChem TSV (cols: smiles, formula, mass, inchi), keep
          rows relevant to the (restricted) query formulas/masses, derive the
          2D-InChIKey from the InChI string (cheap; no mol parsing), dedup by
          2D-InChIKey, and cache the result as a parquet.
  emit  — bucket the cached index by formula and by mass (+-10 ppm), exclude
          the query's own 2D-InChIKey, uniform-random trim to MAX_CANDIDATES-1,
          and write the per-query candidate JSONs (values are raw PubChem
          SMILES; the downstream augment step stereo-strips + dedups them).

Charged/salt PubChem species carry the charge in their formula (``...+``) and
mass, so they neither match a neutral query formula nor fall within +-10 ppm of
a neutral query mass — they are dropped for free, which is the desired
neutral-candidate behaviour.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

MAX_CANDIDATES = 512
PPM_MASS = 10           # +-ppm tolerance for mass buckets (matches build_msg_s4_candidates)
PPM_FILTER = 12         # slightly wider window for the streaming pre-filter (avoid boundary starvation)
CHUNK_ROWS = 5_000_000

DEFAULT_PUBCHEM_TSV = Path("/scratch/project_465003029/data/MassSpecGym/pubchem_inchi.tsv")


# -----------------------------------------------------------------------------
# 2D-InChIKey from InChI (no mol parsing — fast string hash)
# -----------------------------------------------------------------------------

def _worker_init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _ik2d_from_inchi(inchi: str):
    from rdkit.Chem import inchi as rdinchi
    if not isinstance(inchi, str) or not inchi:
        return None
    try:
        ik = rdinchi.InchiToInchiKey(inchi)
        return ik[:14] if ik else None
    except Exception:
        return None


def _parallel_ik2d(inchis: list[str], n_workers: int) -> list:
    if n_workers <= 1:
        return [_ik2d_from_inchi(s) for s in inchis]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        return list(pool.imap(_ik2d_from_inchi, inchis, chunksize=4096))


# -----------------------------------------------------------------------------
# Query loading + short-query restriction
# -----------------------------------------------------------------------------

def _short_keys(json_path: Path, threshold: int) -> set[str]:
    with open(json_path) as f:
        d = json.load(f)
    return {k for k, v in d.items() if len(v) < threshold}


def _load_query_subsets(args):
    """Return (q_formula, q_mass): query rows needing PubChem formula / mass fallback."""
    q = pd.read_parquet(args.queries_parquet)  # cols: inchikey_2d, smiles, formula, exact_mass, fold
    q = q.dropna(subset=["smiles", "formula", "exact_mass", "inchikey_2d"]).reset_index(drop=True)
    if args.restrict_formula_json:
        short_f = _short_keys(args.restrict_formula_json, args.threshold)
        q_formula = q[q["smiles"].isin(short_f)].reset_index(drop=True)
    else:
        q_formula = q
    if args.restrict_mass_json:
        short_m = _short_keys(args.restrict_mass_json, args.threshold)
        q_mass = q[q["smiles"].isin(short_m)].reset_index(drop=True)
    else:
        q_mass = q
    print(f"[queries] total={len(q):,}  formula-fallback={len(q_formula):,}  mass-fallback={len(q_mass):,}")
    return q_formula, q_mass


# -----------------------------------------------------------------------------
# Stage: index
# -----------------------------------------------------------------------------

def _mass_relevant(chunk_mass: np.ndarray, query_masses_sorted: np.ndarray, ppm: float) -> np.ndarray:
    """Boolean mask: chunk rows whose mass is within +-ppm of some query mass."""
    if len(query_masses_sorted) == 0:
        return np.zeros(len(chunk_mass), dtype=bool)
    idx = np.searchsorted(query_masses_sorted, chunk_mass)
    out = np.zeros(len(chunk_mass), dtype=bool)
    tol = chunk_mass * ppm * 1e-6
    for nb in (idx - 1, idx):  # nearest query mass on either side
        valid = (nb >= 0) & (nb < len(query_masses_sorted))
        diff = np.full(len(chunk_mass), np.inf)
        diff[valid] = np.abs(query_masses_sorted[nb[valid]] - chunk_mass[valid])
        out |= diff <= tol
    return out


def _stage_index(args, q_formula, q_mass):
    out_path = args.index_parquet
    if out_path.exists() and not args.force:
        print(f"[index] {out_path} exists; skipping (use --force to rebuild)")
        return
    formulas_needed = set(q_formula["formula"].astype(str))
    masses_needed = np.sort(q_mass["exact_mass"].to_numpy(dtype=float))
    print(f"[index] formulas needed={len(formulas_needed):,}  masses needed={len(masses_needed):,}")
    print(f"[index] streaming {args.pubchem_tsv} (chunks of {CHUNK_ROWS:,}) ...")

    kept_parts: list[pd.DataFrame] = []
    n_seen = n_kept = 0
    t0 = time.perf_counter()
    reader = pd.read_csv(
        args.pubchem_tsv, sep="\t",
        usecols=["smiles", "formula", "mass", "inchi"],
        dtype={"smiles": str, "formula": str, "inchi": str},
        chunksize=CHUNK_ROWS, quoting=csv.QUOTE_NONE, on_bad_lines="skip",
    )
    for ci, chunk in enumerate(reader):
        n_seen += len(chunk)
        mass = pd.to_numeric(chunk["mass"], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(mass)
        f_rel = chunk["formula"].isin(formulas_needed).to_numpy() & valid
        m_rel = np.zeros(len(chunk), dtype=bool)
        m_rel[valid] = _mass_relevant(mass[valid], masses_needed, PPM_FILTER)
        keep = f_rel | m_rel
        if keep.any():
            part = chunk.loc[keep, ["smiles", "formula", "inchi"]].copy()
            part["exact_mass"] = mass[keep]
            kept_parts.append(part)
            n_kept += int(keep.sum())
        if (ci + 1) % 5 == 0:
            print(f"[index]   chunk {ci+1}: seen={n_seen:,} kept={n_kept:,} "
                  f"({time.perf_counter()-t0:.0f}s)")

    if not kept_parts:
        raise SystemExit("[index] no PubChem rows matched any query formula/mass — aborting")
    df = pd.concat(kept_parts, ignore_index=True)
    print(f"[index] kept {len(df):,} / {n_seen:,} rows; computing 2D-InChIKey ...")
    df["inchikey_2d"] = _parallel_ik2d(df["inchi"].tolist(), args.n_workers)
    df = df.dropna(subset=["inchikey_2d"]).drop(columns=["inchi"])
    before = len(df)
    df = df.drop_duplicates("inchikey_2d", keep="first").reset_index(drop=True)
    print(f"[index] {before:,} -> {len(df):,} after 2D-InChIKey dedup")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd")
    print(f"[index] wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB) in {time.perf_counter()-t0:.0f}s")


# -----------------------------------------------------------------------------
# Stage: emit
# -----------------------------------------------------------------------------

def _trim_uniform(rng: random.Random, candidates: list[str], cap: int) -> list[str]:
    if len(candidates) <= cap:
        return candidates
    return rng.sample(candidates, cap)


def _stage_emit(args, q_formula, q_mass):
    pool = pd.read_parquet(args.index_parquet)
    print(f"[emit] pubchem index pool={len(pool):,}")

    # FORMULA buckets.
    formula_to_entries: dict[str, list[tuple[str, str]]] = {}
    for smi, formula, ik in zip(pool["smiles"], pool["formula"], pool["inchikey_2d"]):
        formula_to_entries.setdefault(formula, []).append((smi, ik))
    rng = random.Random(0)
    out_f: dict[str, list[str]] = {}
    for q_smi, q_formula_str, q_ik in zip(q_formula["smiles"], q_formula["formula"], q_formula["inchikey_2d"]):
        bucket = [smi for (smi, ik) in formula_to_entries.get(q_formula_str, []) if ik != q_ik]
        if bucket:
            out_f[q_smi] = _trim_uniform(rng, bucket, MAX_CANDIDATES - 1)
    _write_json(out_f, args.out_formula, "formula")

    # MASS buckets (+-PPM_MASS).
    pool_sorted = pool.sort_values("exact_mass").reset_index(drop=True)
    masses = pool_sorted["exact_mass"].to_numpy(dtype=float)
    smiles_arr = pool_sorted["smiles"].to_numpy()
    iks = pool_sorted["inchikey_2d"].to_numpy()
    rng = random.Random(0)
    out_m: dict[str, list[str]] = {}
    for q_smi, q_mass_val, q_ik in zip(q_mass["smiles"], q_mass["exact_mass"], q_mass["inchikey_2d"]):
        tol = q_mass_val * PPM_MASS * 1e-6
        lo = np.searchsorted(masses, q_mass_val - tol, side="left")
        hi = np.searchsorted(masses, q_mass_val + tol, side="right")
        bucket = [smiles_arr[i] for i in range(lo, hi) if iks[i] != q_ik]
        if bucket:
            out_m[q_smi] = _trim_uniform(rng, bucket, MAX_CANDIDATES - 1)
    _write_json(out_m, args.out_mass, "mass")


def _write_json(out: dict, path: Path, label: str) -> None:
    if not out:
        print(f"[emit_{label}] no queries had PubChem candidates; writing empty JSON")
    sizes = np.asarray([len(v) for v in out.values()]) if out else np.array([0])
    print(f"[emit_{label}] queries with candidates={len(out):,}  "
          f"median={int(np.median(sizes))} mean={sizes.mean():.1f} max={int(sizes.max())}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(out, f)
    print(f"[emit_{label}] wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["index", "emit", "all"])
    p.add_argument("--queries-parquet", type=Path, required=True,
                   help="extract_unique_mols.parquet (cols: smiles, formula, exact_mass, inchikey_2d, fold)")
    p.add_argument("--pubchem-tsv", type=Path, default=DEFAULT_PUBCHEM_TSV)
    p.add_argument("--index-parquet", type=Path, required=True,
                   help="cache path for the query-relevant PubChem index")
    p.add_argument("--out-formula", type=Path, required=True)
    p.add_argument("--out-mass", type=Path, required=True)
    p.add_argument("--restrict-formula-json", type=Path, default=None,
                   help="S4 formula JSON; restrict to queries with < threshold candidates")
    p.add_argument("--restrict-mass-json", type=Path, default=None,
                   help="S4 mass JSON; restrict to queries with < threshold candidates")
    p.add_argument("--threshold", type=int, default=8)
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("WORKERS", 32)))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    q_formula, q_mass = _load_query_subsets(args)
    stages = ["index", "emit"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n=== STAGE: {stage} ===")
        if stage == "index":
            _stage_index(args, q_formula, q_mass)
        else:
            _stage_emit(args, q_formula, q_mass)


if __name__ == "__main__":
    main()
