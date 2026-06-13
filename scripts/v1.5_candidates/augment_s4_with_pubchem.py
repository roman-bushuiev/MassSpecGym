#!/usr/bin/env python
"""Augment MassSpecGym S4 retrieval candidates with PubChem candidates for
queries where the S4 bucket has fewer than 8 entries.

Rule (per Roman):
  - Keep ALL existing S4 candidates as-is (query at index 0, then S4-generated).
  - Append PubChem candidates whose 2D-InChIKey is NOT already present in the
    S4 list. Append in PubChem's existing order.
  - Stop once total list length reaches 1024.
  - Apply only to queries where the original S4 list length < 8.

Source PubChem: ``MassSpecGym1.5_retrieval_candidates_{formula,mass}.json``
(re-canonicalised v1.5 release; keys align with S4 JSONs).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

DATA = Path(os.environ.get(
    "MSG_DATA_DIR",
    "/pfs/lustrep2/scratch/project_465003029/rbushuie/DreaMS-Mol_dev/MassSpecGym/data",
))

S4_FORMULA  = DATA / "MassSpecGym_S4_retrieval_candidates_formula_nostereo.json"
S4_MASS     = DATA / "MassSpecGym_S4_retrieval_candidates_mass_nostereo.json"
PC_FORMULA  = DATA / "MassSpecGym1.5_retrieval_candidates_formula.json"
PC_MASS     = DATA / "MassSpecGym1.5_retrieval_candidates_mass.json"

OUT_FORMULA = DATA / "MassSpecGym_S4plusPC_retrieval_candidates_formula_nostereo.json"
OUT_MASS    = DATA / "MassSpecGym_S4plusPC_retrieval_candidates_mass_nostereo.json"

THRESHOLD = 8       # augment when len(s4_list) < this
CAP = 512           # final list size cap


# Worker pool globals.
_WORKER_SMILES_TO_IK: dict[str, str | None] = {}


def _worker_init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _worker_strip_and_ik(smi: str):
    """Return (smi, stripped_smi_or_None, ik2d_or_None) for a SMILES.

    2D-InChIKey is stereo-invariant so it can be derived from either the input
    or the stripped molecule; we use the input for fewer steps.
    """
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi, None, None
    try:
        stripped = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except Exception:
        stripped = None
    try:
        ik = Chem.MolToInchiKey(m)
        ik2d = ik[:14] if ik else None
    except Exception:
        ik2d = None
    return smi, stripped, ik2d


def _resolve_set(smiles_set: set[str], n_workers: int) -> dict[str, tuple[str | None, str | None]]:
    """Map ``{raw_smi -> (stripped_smi, ik2d)}`` for a set of SMILES."""
    if not smiles_set:
        return {}
    import multiprocessing as mp
    items = list(smiles_set)
    if n_workers <= 1:
        return {smi: (st, ik) for smi, st, ik in (_worker_strip_and_ik(s) for s in items)}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        return {smi: (st, ik) for smi, st, ik in pool.imap_unordered(_worker_strip_and_ik, items, chunksize=512)}


def _augment(label: str, s4_path: Path, pc_path: Path, out_path: Path, n_workers: int) -> dict:
    print(f"\n=== {label} ===")
    t0 = time.perf_counter()
    with s4_path.open() as f:
        s4 = json.load(f)
    print(f"[load] S4 {s4_path.name}: {len(s4):,} keys in {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    with pc_path.open() as f:
        pc_raw = json.load(f)
    print(f"[load] PubChem v1.5 {pc_path.name}: {len(pc_raw):,} keys in {time.perf_counter()-t0:.1f}s")

    # PubChem v1.5 keys are stereo-bearing while S4 keys are nostereo — re-key
    # PubChem by the stripped form so we can look up by S4 keys.
    from massspecgym.utils import _strip_stereo_parallel
    print(f"[align] stereo-stripping PubChem keys ({len(pc_raw):,}) for alignment ...")
    t0 = time.perf_counter()
    pc_key_map = _strip_stereo_parallel(set(pc_raw.keys()), n_workers=n_workers)
    pc: dict[str, list[str]] = {}
    for q, lst in pc_raw.items():
        q_no = pc_key_map.get(q)
        if not q_no:
            continue
        # If multiple PubChem keys collapse to the same nostereo key, union the lists
        # (preserving the first occurrence's order).
        if q_no in pc:
            seen = set(pc[q_no])
            for c in lst:
                if c not in seen:
                    pc[q_no].append(c)
                    seen.add(c)
        else:
            pc[q_no] = list(lst)
    print(f"[align]   pc rekeyed in {time.perf_counter()-t0:.1f}s; {len(pc):,} keys after collapse")

    # Identify queries needing augmentation.
    short_keys = [q for q in s4 if len(s4[q]) < THRESHOLD]
    short_in_pc = [q for q in short_keys if q in pc]
    print(f"[scan] queries with S4 len < {THRESHOLD}: {len(short_keys):,}")
    print(f"[scan]   ...of which present in PubChem v1.5 (post-align): {len(short_in_pc):,}")

    # S4 candidates are already stereo-stripped; PubChem v1.5 candidates are not.
    # We need (stripped_smi, ik2d) per SMILES so the merged list is fully nostereo.
    smiles_to_resolve: set[str] = set()
    for q in short_in_pc:
        smiles_to_resolve.update(s4[q])
        smiles_to_resolve.update(pc[q][:CAP])  # don't bother beyond the cap
    print(f"[resolve] computing (stripped, 2D-InChIKey) for {len(smiles_to_resolve):,} unique SMILES "
          f"using {n_workers} workers ...")
    t0 = time.perf_counter()
    smi2info = _resolve_set(smiles_to_resolve, n_workers)
    n_ok_ik = sum(1 for st, ik in smi2info.values() if ik)
    print(f"[resolve]   done in {time.perf_counter()-t0:.1f}s; "
          f"ik2d_ok={n_ok_ik:,}  ik2d_fail={len(smi2info)-n_ok_ik:,}")

    # Build augmented JSON. Copy untouched keys verbatim; augment short ones.
    out: dict[str, list[str]] = {}
    n_aug = n_added_total = 0
    list_size_before = list_size_after = 0
    for q, cands in s4.items():
        if len(cands) >= THRESHOLD or q not in pc:
            out[q] = cands
            continue
        existing_iks = {smi2info.get(s, (None, None))[1] for s in cands}
        existing_iks.discard(None)
        merged = list(cands)
        merged_set = set(merged)
        for c in pc[q]:
            if len(merged) >= CAP:
                break
            stripped, ik = smi2info.get(c, (None, None))
            if not ik or not stripped or ik in existing_iks or stripped in merged_set:
                continue
            merged.append(stripped)
            merged_set.add(stripped)
            existing_iks.add(ik)
        n_added = len(merged) - len(cands)
        if n_added:
            n_aug += 1
            n_added_total += n_added
        list_size_before += len(cands)
        list_size_after += len(merged)
        out[q] = merged

    print(f"[aug] queries augmented: {n_aug:,} / {len(short_in_pc):,}")
    print(f"[aug] total PubChem candidates added: {n_added_total:,}")
    if n_aug:
        print(f"[aug] mean list size: before={list_size_before/len(short_in_pc):.1f} "
              f"after={list_size_after/len(short_in_pc):.1f}")

    # Stats on the full output JSON.
    import numpy as np
    L = np.asarray([len(v) for v in out.values()])
    n_trivial = int((L == 1).sum())
    n_lt8 = int((L < THRESHOLD).sum())
    print(f"[stats] OUTPUT — n_queries={len(out):,} "
          f"median={int(np.median(L))} mean={L.mean():.1f} p90={int(np.percentile(L,90))} max={int(L.max())}")
    print(f"[stats] trivial (len=1): {n_trivial:,} ({n_trivial/len(out)*100:.2f}%)")
    print(f"[stats] len<{THRESHOLD}: {n_lt8:,} ({n_lt8/len(out)*100:.2f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[write] wrote {out_path} ({size_mb:.1f} MB)")

    return {
        "label": label,
        "n_queries": len(out),
        "n_short_input": len(short_keys),
        "n_short_in_pc": len(short_in_pc),
        "n_augmented": n_aug,
        "n_pc_added_total": n_added_total,
        "median_after": int(np.median(L)),
        "mean_after": float(L.mean()),
        "trivial_after": n_trivial,
        "lt8_after": n_lt8,
    }


def main():
    global THRESHOLD, CAP
    p = argparse.ArgumentParser()
    p.add_argument("--s4-formula", type=Path, default=S4_FORMULA)
    p.add_argument("--s4-mass", type=Path, default=S4_MASS)
    p.add_argument("--pc-formula", type=Path, default=PC_FORMULA)
    p.add_argument("--pc-mass", type=Path, default=PC_MASS)
    p.add_argument("--out-formula", type=Path, default=OUT_FORMULA)
    p.add_argument("--out-mass", type=Path, default=OUT_MASS)
    p.add_argument("--threshold", type=int, default=THRESHOLD)
    p.add_argument("--cap", type=int, default=CAP)
    p.add_argument("--only", choices=["formula", "mass", "both"], default="both")
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("WORKERS", 32)))
    args = p.parse_args()
    THRESHOLD, CAP = args.threshold, args.cap

    results = []
    if args.only in ("formula", "both"):
        results.append(_augment("FORMULA", args.s4_formula, args.pc_formula, args.out_formula, args.n_workers))
    if args.only in ("mass", "both"):
        results.append(_augment("MASS", args.s4_mass, args.pc_mass, args.out_mass, args.n_workers))

    print("\n=== SUMMARY ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
