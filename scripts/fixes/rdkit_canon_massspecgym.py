"""Canonicalise MassSpecGym SMILES + recompute all SMILES-derived columns.

This is the first step of the published v1.5 build pipeline:

    1. rdkit_canon_massspecgym.py     ← canonicalise + recompute derived cols
    2. mine retrieval candidates      ← S4 build pipeline (separate)

Reads:
  - MassSpecGym/data/MassSpecGym.tsv   (PubChem-standardised release)

Writes:
  - MassSpecGym/data/v1.5/MassSpecGym1.5.tsv
  - MassSpecGym/data/v1.5/MassSpecGym1.5.mgf

For every row:
  - smiles              ← RDKit canonical, stereo-stripped
                           (massspecgym.utils.rdkit_canonical_smiles)
  - formula             ← rdMolDescriptors.CalcMolFormula(mol)
  - inchikey            ← first 14 chars of MolToInchiKey(MolToInchi(mol))
                           (the 2D-InChIKey convention used in v1)
  - parent_mass         ← rdMolDescriptors.CalcExactMolWt(mol)
  - precursor_formula   ← Hill-notation formula of (n_parents * formula
                           +/- adduct ion components), via matchms
                           adduct parsing

Columns NOT recomputed (passed through unchanged):
  - identifier, mzs, intensities         (measurement / identity)
  - precursor_mz                          (measured m/z, not derived)
  - adduct, instrument_type,
    collision_energy, fold,
    simulation_challenge                  (annotation / metadata)

Logs per-column change counts (rows where the recomputed value differs
from the value carried over from the input TSV).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.exporting import save_as_mgf
from matchms.filtering.filter_utils.interpret_unknown_adduct import (
    get_ions_from_adduct,
    split_ion,
)
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
from tqdm import tqdm

import massspecgym.utils as utils

RDLogger.logger().setLevel(RDLogger.CRITICAL)

BASE = Path("/scratch/project_465002061/rbushuie/DreaMS-Mol_dev")
TSV_IN = BASE / "MassSpecGym/data/MassSpecGym.tsv"
OUT_DIR = BASE / "MassSpecGym/data/v1.5"
TSV_OUT = OUT_DIR / "MassSpecGym1.5.tsv"
MGF_OUT = OUT_DIR / "MassSpecGym1.5.mgf"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ─── Formula arithmetic (verbatim from notebooks/dataset_construction/2_clean_library.ipynb)
class Formula:
    def __init__(self, formula: str):
        self.dict_representation = self.get_atom_and_counts(formula)

    @staticmethod
    def get_atom_and_counts(formula: str) -> dict[str, int]:
        parts = re.findall(r"[A-Z][a-z]?|[0-9]+", formula)
        out: dict[str, int] = {}
        for i, atom in enumerate(parts):
            if atom.isnumeric():
                continue
            mult = int(parts[i + 1]) if len(parts) > i + 1 and parts[i + 1].isnumeric() else 1
            out[atom] = out.get(atom, 0) + mult
        return out

    def __add__(self, other: "Formula") -> "Formula":
        new = Formula("")
        new.dict_representation = self.dict_representation.copy()
        for a, v in other.dict_representation.items():
            new.dict_representation[a] = new.dict_representation.get(a, 0) + v
        return new

    def __sub__(self, other: "Formula") -> "Formula | None":
        new = Formula("")
        new.dict_representation = self.dict_representation.copy()
        for a, v in other.dict_representation.items():
            if a not in new.dict_representation:
                return None
            new.dict_representation[a] -= v
            if new.dict_representation[a] < 0:
                return None
        return new

    def __str__(self) -> str:
        d = self.dict_representation
        c = d.get("C", 0); h = d.get("H", 0)
        rest = sorted((k, v) for k, v in d.items() if k not in ("C", "H"))
        out = ""
        if c > 0:
            out += "C" + (str(c) if c > 1 else "")
        if h > 0:
            out += "H" + (str(h) if h > 1 else "")
        for k, v in rest:
            out += k + (str(v) if v > 1 else "")
        return out


def precursor_formula_from(formula: str, adduct: str) -> str | None:
    """Replicates notebooks/dataset_construction/2_clean_library.ipynb::add_precursor_formula
    (kept byte-identical for v1.5)."""
    try:
        n_parents, ions = get_ions_from_adduct(adduct)
    except Exception:
        return None
    if formula is None:
        return None
    parent = Formula(formula)
    out = Formula("")
    for _ in range(n_parents):
        out = out + parent
    for ion in ions:
        sign, number, ion_formula = split_ion(ion)
        for _ in range(number):
            if sign == "+":
                out = out + Formula(ion_formula)
            elif sign == "-":
                tmp = out - Formula(ion_formula)
                if tmp is None:
                    return None
                out = tmp
    return str(out)


canon_counts: Counter = Counter()


def rdkit_canonical(smi: str) -> str:
    """RDKit canonical SMILES, stereo stripped (v1.5 convention).

    Equivalent to ``Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)``,
    with the same outcome-bookkeeping fallback as ``utils.rdkit_canonical_smiles``.
    The stereo strip is what makes the SMILES join key match the
    candidate JSON keys (which are built stereo-stripped).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        canon_counts["kept_original"] += 1
        return smi
    try:
        result = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        canon_counts["canonical_true"] += 1
        return result
    except Exception:
        try:
            result = Chem.MolToSmiles(mol, isomericSmiles=False)
            canon_counts["canonical_fallback"] += 1
            return result
        except Exception:
            canon_counts["kept_original"] += 1
            return smi


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Reading %s ...", TSV_IN)
    df = pd.read_csv(TSV_IN, sep="\t")
    log.info("Loaded %d rows × %d cols: %s", len(df), df.shape[1], list(df.columns))

    # ── 1. Canonicalise SMILES ────────────────────────────────────────────
    unique_smiles = sorted(df["smiles"].unique())
    log.info("Canonicalising %d unique SMILES ...", len(unique_smiles))
    canon_map = {s: rdkit_canonical(s) for s in tqdm(unique_smiles, desc="Canon SMILES")}
    n_unique_changed = sum(1 for k, v in canon_map.items() if k != v)
    log.info("  MolToSmiles(canonical=True):     %d", canon_counts["canonical_true"])
    log.info("  MolToSmiles() fallback:          %d", canon_counts["canonical_fallback"])
    log.info("  Kept original SMILES:            %d", canon_counts["kept_original"])
    log.info("  Unique SMILES that changed:      %d / %d", n_unique_changed, len(unique_smiles))

    orig_cols = df[["smiles", "formula", "inchikey", "parent_mass", "precursor_formula"]].copy()
    df["smiles"] = df["smiles"].map(canon_map)

    # ── 2. Recompute SMILES-derived columns row-by-row ────────────────────
    new_formula: list[str | None] = [None] * len(df)
    new_inchikey: list[str | None] = [None] * len(df)
    new_parent_mass: list[float | None] = [None] * len(df)
    new_precursor_formula: list[str | None] = [None] * len(df)
    err_counts: Counter = Counter()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Recompute"):
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            err_counts["parse_failed"] += 1
            new_formula[idx] = row["formula"]
            new_inchikey[idx] = row["inchikey"]
            new_parent_mass[idx] = row["parent_mass"]
            new_precursor_formula[idx] = row["precursor_formula"]
            continue

        f = rdMolDescriptors.CalcMolFormula(mol)
        m = float(rdMolDescriptors.CalcExactMolWt(mol))
        inchi = MolToInchi(mol)
        ik = InchiToInchiKey(inchi)[:14] if inchi else None
        if ik is None:
            err_counts["inchikey_compute_failed"] += 1
        pf = precursor_formula_from(f, row["adduct"])
        if pf is None:
            err_counts["precursor_formula_compute_failed"] += 1
            pf = row["precursor_formula"]

        new_formula[idx] = f
        new_inchikey[idx] = ik if ik is not None else row["inchikey"]
        new_parent_mass[idx] = m
        new_precursor_formula[idx] = pf

    df["formula"] = new_formula
    df["inchikey"] = new_inchikey
    df["parent_mass"] = new_parent_mass
    df["precursor_formula"] = new_precursor_formula

    # ── 3. Per-column change report ───────────────────────────────────────
    log.info("--- Per-column change counts vs input TSV ---")
    MASS_TOL = 1e-3
    n_total = len(df)

    def _str_diff(a: pd.Series, b: pd.Series) -> int:
        sa = a.astype(object).where(a.notna(), "<NA>").astype(str)
        sb = b.astype(object).where(b.notna(), "<NA>").astype(str)
        return int((sa != sb).sum())

    n_smiles_diff = _str_diff(orig_cols["smiles"], df["smiles"])
    n_formula_diff = _str_diff(orig_cols["formula"], df["formula"])
    n_inchikey_diff = _str_diff(orig_cols["inchikey"], df["inchikey"])
    n_precursor_formula_diff = _str_diff(orig_cols["precursor_formula"], df["precursor_formula"])
    diff_mass = (~np.isclose(
        orig_cols["parent_mass"].astype(float).fillna(-1e30),
        df["parent_mass"].astype(float).fillna(-1e30),
        atol=MASS_TOL, rtol=0,
    ))
    n_parent_mass_diff = int(diff_mass.sum())

    for name, n in [
        ("smiles", n_smiles_diff),
        ("formula", n_formula_diff),
        ("inchikey", n_inchikey_diff),
        ("parent_mass", n_parent_mass_diff),
        ("precursor_formula", n_precursor_formula_diff),
    ]:
        log.info("  %-18s %7d / %d  (%.3f %%)", name, n, n_total, 100 * n / n_total)

    if err_counts:
        log.info("--- Per-row computation errors ---")
        for k, v in err_counts.most_common():
            log.info("  %-30s %d", k, v)
    else:
        log.info("  No row-level computation errors.")

    # ── 4. Write TSV ──────────────────────────────────────────────────────
    log.info("Writing %s ...", TSV_OUT)
    df.to_csv(TSV_OUT, sep="\t", index=False)

    # ── 5. Write MGF (one-pass conversion from the TSV we just wrote) ─────
    log.info("Building MGF from %s ...", TSV_OUT)
    spectra: list[Spectrum] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Build spectra"):
        metadata = {
            k: v for k, v in row.items()
            if k not in ("mzs", "intensities") and v is not np.nan
        }
        spectra.append(Spectrum(
            mz=utils.parse_spec_array(row["mzs"]),
            intensities=utils.parse_spec_array(row["intensities"]),
            metadata=metadata,
        ))
    if MGF_OUT.exists():
        log.info("  Removing existing %s", MGF_OUT)
        os.remove(MGF_OUT)
    log.info("Writing %s (%d spectra) ...", MGF_OUT, len(spectra))
    save_as_mgf(spectra, str(MGF_OUT))

    log.info("=" * 60)
    log.info("DONE.")
    log.info("  TSV: %s", TSV_OUT)
    log.info("  MGF: %s", MGF_OUT)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
