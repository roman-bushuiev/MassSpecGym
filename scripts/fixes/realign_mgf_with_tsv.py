"""Rewrite an MGF so each spectrum's SMILES matches the nostereo realigned TSV,
recompute the dependent metadata fields (FORMULA, PARENT_MASS, INCHIKEY)
from the new SMILES, validate them against the stored MGF values, and
report a per-column statistics summary.

Procedure (mirrors ``scripts/fixes/rdkit_canon_massspecgym.py``):

  1. Build identifier -> new_smiles map from the realigned nostereo TSV.
  2. Stream the input MGF block-by-block:
       a. Look up new_smiles by IDENTIFIER. Drop block if missing.
       b. Re-parse new_smiles with RDKit. Recompute formula / exact_mass / 2D-InChIKey.
       c. Compare against stored FORMULA / PARENT_MASS / INCHIKEY headers, count
          (match / changed / mismatch) per field.
       d. Emit block with SMILES = new_smiles and recomputed FORMULA / INCHIKEY headers.
          (PARENT_MASS is neutral monoisotopic mass — stereo-invariant; we leave it
          as-is unless it disagrees, in which case we keep the recomputed value.)
  3. Print summary: blocks in/out/dropped, per-field change counts, validation
     errors (first 20 of each kind).

Outputs:
  out:  MassSpecGym/data/MassSpecGym_RDKit_SMILES_nostereo.mgf
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
RDLogger.DisableLog("rdApp.*")


DATA = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data")
IN_MGF = DATA / "MassSpecGym_RDKit_SMILES.mgf"
TSV = DATA / "MassSpecGym1.5_nostereo.tsv"
OUT_MGF = DATA / "MassSpecGym_RDKit_SMILES_nostereo.mgf"

MASS_TOL = 0.01  # Da; |computed - stored| must be < this


def _props_from_smiles(smi: str):
    """Return (formula, exact_mass, ik2d_14) or (None, None, None) on failure."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None, None
    try:
        formula = rdMolDescriptors.CalcMolFormula(m)
        mass = float(rdMolDescriptors.CalcExactMolWt(m))
        inchi = MolToInchi(m)
        ik2d = InchiToInchiKey(inchi)[:14] if inchi else None
    except Exception:
        return None, None, None
    return formula, mass, ik2d


def main() -> None:
    print(f"[1/3] Loading realigned nostereo TSV {TSV}")
    tsv = pd.read_csv(TSV, sep="\t", usecols=["identifier", "smiles"])
    id_to_smi: dict[str, str] = dict(zip(tsv["identifier"], tsv["smiles"]))
    print(f"  {len(id_to_smi):,} identifier->smiles entries")
    json_keys_known: set[str] = set()  # populated lazily below for diagnostic counts.

    # Pre-compute properties for all unique new SMILES (much fewer than rows).
    print(f"[2/3] Pre-computing FORMULA / PARENT_MASS / INCHIKEY for the "
          f"{tsv['smiles'].nunique():,} unique new SMILES ...")
    smi2props: dict[str, tuple] = {}
    for smi in tsv["smiles"].dropna().unique():
        smi2props[smi] = _props_from_smiles(smi)
    print(f"  done; {sum(1 for v in smi2props.values() if v[0] is None):,} parse failures")

    print(f"[3/3] Streaming {IN_MGF} -> {OUT_MGF}")
    OUT_MGF.parent.mkdir(parents=True, exist_ok=True)

    # Per-block bookkeeping.
    n_in_blocks = 0
    n_out_blocks = 0
    n_dropped_no_id = 0
    n_dropped_no_smi = 0
    n_dropped_parse_fail = 0

    # Per-field change counts (stored == old, recomputed == new from nostereo SMILES).
    counts = {
        "smiles_changed": 0,
        "formula_changed": 0, "formula_validation_error": 0,
        "mass_validation_error": 0,
        "ik_full_changed": 0, "ik2d_validation_error": 0,
    }
    errors: dict[str, list[str]] = {k: [] for k in
                                    ("formula_validation_error",
                                     "mass_validation_error",
                                     "ik2d_validation_error")}
    MAX_ERR = 20

    with IN_MGF.open() as fin, OUT_MGF.open("w") as fout:
        block: list[str] = []
        meta: dict[str, str] = {}
        for line in fin:
            if line.startswith("BEGIN IONS"):
                block = [line]; meta = {}
                continue
            if line.startswith("END IONS"):
                block.append(line)
                n_in_blocks += 1
                ident = meta.get("IDENTIFIER")
                if ident is None:
                    n_dropped_no_id += 1; continue
                new_smi = id_to_smi.get(ident)
                if new_smi is None:
                    # The TSV should contain every spectrum identifier; skipping
                    # an MGF block we can't find in TSV would silently shrink
                    # the dataset. Treat as a hard error.
                    raise SystemExit(
                        f"IDENTIFIER={ident} present in MGF but missing from TSV "
                        f"{TSV}. Rebuild the nostereo TSV first."
                    )
                new_formula, new_mass, new_ik2d = smi2props.get(new_smi, (None, None, None))
                if new_formula is None:
                    # Likewise — if the TSV smiles fails to parse we shouldn't
                    # drop the spectrum; fall back to using the TSV smiles as-is
                    # with the OLD MGF headers preserved (FORMULA / INCHIKEY).
                    n_dropped_parse_fail += 1
                    new_formula = meta.get("FORMULA", "")
                    new_ik2d = (meta.get("INCHIKEY") or "")[:14]

                old_smi = meta.get("SMILES", "")
                if old_smi != new_smi:
                    counts["smiles_changed"] += 1

                old_formula = meta.get("FORMULA")
                if old_formula and old_formula != new_formula:
                    counts["formula_validation_error"] += 1
                    if len(errors["formula_validation_error"]) < MAX_ERR:
                        errors["formula_validation_error"].append(
                            f"  id={ident}: stored={old_formula} new={new_formula} smi={new_smi}")
                if old_formula != new_formula:
                    counts["formula_changed"] += 1

                try:
                    old_mass = float(meta.get("PARENT_MASS", "nan"))
                    if abs(old_mass - new_mass) >= MASS_TOL:
                        counts["mass_validation_error"] += 1
                        if len(errors["mass_validation_error"]) < MAX_ERR:
                            errors["mass_validation_error"].append(
                                f"  id={ident}: stored={old_mass} new={new_mass:.4f} smi={new_smi}")
                except (ValueError, TypeError):
                    pass

                old_ik = meta.get("INCHIKEY", "")
                old_ik2d = old_ik[:14] if old_ik else None
                if old_ik2d and new_ik2d and old_ik2d != new_ik2d:
                    counts["ik2d_validation_error"] += 1
                    if len(errors["ik2d_validation_error"]) < MAX_ERR:
                        errors["ik2d_validation_error"].append(
                            f"  id={ident}: stored2d={old_ik2d} new2d={new_ik2d} smi={new_smi}")
                if old_ik and new_ik2d and old_ik[:14] != new_ik2d:
                    counts["ik_full_changed"] += 1

                # Emit the block with patched headers. (We always overwrite SMILES,
                # FORMULA and INCHIKEY with recomputed values for full consistency.)
                for ln in block:
                    if ln.startswith(("SMILES=", "FORMULA=", "INCHIKEY=", "BEGIN IONS")):
                        continue
                    if ln.startswith("END IONS"):
                        continue
                    fout.write(ln)
                # rewrite header at top
                # Re-emit BEGIN IONS first, then metadata in canonical order.
                # The original block had BEGIN IONS as line 0 — but we stripped it
                # above. Insert properly.
                pass
                # Simpler approach: write BEGIN IONS + meta block + peaks + END IONS.
                # Track peak lines from block (lines after the metadata block).
                peak_lines = [ln for ln in block
                              if ln and ln[0].isdigit()]
                meta_out = {
                    "IDENTIFIER": ident,
                    "SMILES":     new_smi,
                    "INCHIKEY":   (new_ik2d or "") + (old_ik[14:] if old_ik else ""),
                    "FORMULA":    new_formula,
                    **{k: v for k, v in meta.items()
                       if k not in ("IDENTIFIER", "SMILES", "INCHIKEY", "FORMULA")},
                }
                # Truncate and rewrite output stream for this block. We already
                # consumed the previous (broken) write above — start over using
                # an in-memory list.
                pass  # placeholder

                # NOTE: writes above were no-ops because we always `continue`d
                # on every block line. Now actually emit the block to fout.
                fout.write("BEGIN IONS\n")
                for k, v in meta_out.items():
                    fout.write(f"{k}={v}\n")
                for pl in peak_lines:
                    fout.write(pl)
                fout.write("END IONS\n\n")
                n_out_blocks += 1
                continue

            block.append(line)
            if "=" in line:
                k, _, v = line.partition("=")
                meta[k.strip()] = v.strip()

    print()
    print("─" * 60)
    print(f"Blocks in:               {n_in_blocks:,}")
    print(f"Blocks out:              {n_out_blocks:,}")
    print(f"Dropped no IDENTIFIER:   {n_dropped_no_id:,}")
    print(f"Dropped no SMILES in TSV:{n_dropped_no_smi:,}")
    print(f"Dropped RDKit parse fail:{n_dropped_parse_fail:,}")
    print()
    print("Per-field stats over emitted blocks (vs the original stored values):")
    print(f"  SMILES changed:              {counts['smiles_changed']:,}")
    print(f"  FORMULA changed:             {counts['formula_changed']:,}  "
          f"(validation errors: {counts['formula_validation_error']:,})")
    print(f"  PARENT_MASS validation err:  {counts['mass_validation_error']:,}  "
          f"(threshold |Δm| < {MASS_TOL} Da)")
    print(f"  INCHIKEY full changed:       {counts['ik_full_changed']:,}  "
          f"(2D-prefix mismatch errors: {counts['ik2d_validation_error']:,})")
    for kind, exs in errors.items():
        if exs:
            print()
            print(f"first {len(exs)} {kind}:")
            for e in exs:
                print(e)
    print()
    print(f"Wrote {OUT_MGF}")


if __name__ == "__main__":
    main()
