"""Realign MassSpecGym1.5_nostereo.tsv 'smiles' column with the candidate JSON keys.

The build pipeline picks one representative SMILES per 2D-InChIKey (via
lex-min on the stereo-bearing TSV smiles); strip_stereo_tsv strips per row.
For ~18 InChIKeys there are multiple post-strip SMILES strings, and the JSON
only contains one of them. This script rewrites the TSV 'smiles' so that
every spectrum row uses the JSON's representative SMILES for its 2D-InChIKey.

Idempotent. Reads ``MassSpecGym1.5_nostereo.tsv`` + the formula JSON.
Writes back to the same TSV path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

DATA = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data")
TSV_PATH = DATA / "MassSpecGym1.5_nostereo.tsv"
JSON_PATH = DATA / "MassSpecGym_S4_retrieval_candidates_formula_nostereo.json"


def _ik2d(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


def main() -> None:
    print(f"Loading JSON {JSON_PATH}")
    with JSON_PATH.open() as f:
        js = json.load(f)
    print(f"  {len(js):,} keys")

    print("Building ik2d -> JSON_key map")
    ik2d_to_key: dict[str, str] = {}
    n_dup = 0
    for k in js:
        ik = _ik2d(k)
        if not ik:
            continue
        if ik in ik2d_to_key and ik2d_to_key[ik] != k:
            n_dup += 1
        else:
            ik2d_to_key[ik] = k
    print(f"  {len(ik2d_to_key):,} unique ik2d (dup={n_dup})")

    print(f"Loading TSV {TSV_PATH}")
    tsv = pd.read_csv(TSV_PATH, sep="\t")
    print(f"  {len(tsv):,} rows; columns: {list(tsv.columns)}")
    n_unique_smi_before = tsv["smiles"].nunique()

    # Map each row's 2D-InChIKey to the JSON's representative SMILES.
    tsv["ik2d"] = tsv["inchikey"].str[:14]
    tsv["smiles_new"] = tsv["ik2d"].map(ik2d_to_key)
    n_mapped = int(tsv["smiles_new"].notna().sum())
    n_changed = int((tsv["smiles_new"].notna() & (tsv["smiles_new"] != tsv["smiles"])).sum())
    n_unmapped = int(tsv["smiles_new"].isna().sum())
    print(f"  mapped={n_mapped:,} changed={n_changed:,} unmapped={n_unmapped:,}")
    if n_unmapped:
        ex = tsv.loc[tsv["smiles_new"].isna(), "smiles"].dropna().unique()[:5]
        print(f"  unmapped smiles examples (first 5): {list(ex)}")

    # Replace SMILES with the JSON-canonical form when the ik2d is mapped; otherwise
    # keep the original (per-row stripped) SMILES. We do NOT drop any rows — the
    # MassSpecGym dataset preserves every spectrum even when its query SMILES has no
    # candidate JSON entry (downstream retrieval code skips such queries gracefully).
    tsv["smiles"] = tsv["smiles_new"].combine_first(tsv["smiles"])
    tsv = tsv.drop(columns=["smiles_new", "ik2d"])
    n_in_json = int(tsv["smiles"].isin(set(js.keys())).sum())
    n_not_in_json = len(tsv) - n_in_json
    print(f"  rows whose final smiles IS a JSON key:  {n_in_json:,}")
    print(f"  rows whose final smiles is NOT in JSON: {n_not_in_json:,}  (kept as-is, no candidates for retrieval)")
    print(f"  unique smiles: {n_unique_smi_before:,} -> {tsv['smiles'].nunique():,}")

    tmp = TSV_PATH.with_suffix(".tsv.tmp")
    tsv.to_csv(tmp, sep="\t", index=False)
    tmp.rename(TSV_PATH)
    print(f"Wrote {TSV_PATH}")


if __name__ == "__main__":
    main()
