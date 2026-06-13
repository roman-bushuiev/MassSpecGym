"""Re-canonicalise every candidate-value SMILES in the v1.5 JSON files.

Step 5 of the published v1.5 build pipeline. The earlier mining stages
(emit / pubchem_aug / molpher_aug) produce candidate values that are
mostly — but not exclusively — in RDKit canonical + stereo-stripped form.
A small fraction (~0.005-0.02 % of unique candidates, traceable to the
Molpher RerouteBond augmentation which emits explicit-H notation) is
chemically equivalent to the canonical form but written with a different
string.

This script reads each v1.5 candidate JSON, applies
``Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)`` to every
unique candidate value in parallel, rewrites the JSON values in place,
and verifies the result. The query stays at index 0 and the per-bucket
cap is unchanged.

Within-bucket 2D-InChIKey uniqueness is preserved by construction:
canonicalisation maps each input SMILES to a single canonical string per
molecule, so two values that produced the same molecule before will
produce the same string after, and ``dict`` lookup via dedup-while-rewriting
removes the duplicates without needing a separate 2D-IK computation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.logger().setLevel(RDLogger.CRITICAL)

BASE = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev")
V15  = BASE / "MassSpecGym/data/v1.5"

N_WORKERS = max(1, (os.cpu_count() or 8) - 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _canon_worker(smi: str) -> tuple[str, str | None]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, None
    try:
        return smi, Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return smi, None


def process_json(path: Path) -> None:
    log.info("=" * 60)
    log.info("Reading %s", path)
    cands: dict[str, list[str]] = json.load(path.open())
    log.info("  %d keys", len(cands))

    all_smiles: set[str] = set()
    for vals in cands.values():
        all_smiles.update(vals)
    log.info("  %d unique candidate SMILES across all lists", len(all_smiles))

    log.info("Canonicalising in parallel (%d workers) ...", N_WORKERS)
    canon_map: dict[str, str | None] = {}
    with Pool(N_WORKERS) as pool:
        for smi, c in tqdm(
            pool.imap_unordered(_canon_worker, sorted(all_smiles), chunksize=4000),
            total=len(all_smiles),
        ):
            canon_map[smi] = c
    n_changed = sum(1 for s, c in canon_map.items() if c is not None and c != s)
    n_none = sum(1 for c in canon_map.values() if c is None)
    log.info("  Changed:        %d / %d (%.5f %%)",
             n_changed, len(all_smiles), 100 * n_changed / len(all_smiles))
    log.info("  Parse failures: %d", n_none)

    log.info("Rewriting bucket lists (canon + dedup, query at index 0, cap preserved) ...")
    stats = Counter()
    new_cands: dict[str, list[str]] = {}
    for key, vals in tqdm(cands.items(), total=len(cands)):
        seen: set[str] = set()
        out: list[str] = []
        # Key is itself canonical by upstream invariant (verified separately);
        # use it verbatim so JSON lookup by TSV smiles remains exact.
        out.append(key); seen.add(key)
        for v in vals:
            cv = canon_map.get(v, v) or v
            if cv in seen:
                stats["dedup_dropped"] += 1
                continue
            seen.add(cv)
            out.append(cv)
        new_cands[key] = out
        stats["total_keys"] += 1
        stats["final_total_len"] += len(out)

    log.info("--- Rewrite summary ---")
    for k, v in stats.items():
        log.info("  %-25s %d", k, v)
    log.info("  mean list length after rewrite: %.2f",
             stats["final_total_len"] / max(stats["total_keys"], 1))

    bad_first = sum(1 for k, v in new_cands.items() if not v or v[0] != k)
    log.info("  Keys whose first element != key: %d", bad_first)

    log.info("Writing %s ...", path)
    with path.open("w") as f:
        json.dump(new_cands, f)
    log.info("  written; size %.1f MB", path.stat().st_size / 1e6)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--formula-json", type=Path,
                   default=V15 / "MassSpecGym1.5_retrieval_candidates_formula.json")
    p.add_argument("--mass-json", type=Path,
                   default=V15 / "MassSpecGym1.5_retrieval_candidates_mass.json")
    args = p.parse_args()
    for path in (args.formula_json, args.mass_json):
        process_json(path)
    log.info("=" * 60)
    log.info("DONE.")


if __name__ == "__main__":
    main()
