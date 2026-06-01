"""Replace candidate SMILES in MassSpecGym retrieval JSON with canonical PubChem
SMILES, matched first by InChI then by InChIKey as fallback.

Usage:
    python make_pubchem_cands_canonical.py <input_json>

Writes <input_json stem>_canon2.json in the same directory.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

PUBCHEM_TSV = "/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data/pubchem_inchi.tsv"
MOLECULES_4M = "/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/data/massspecgym/MassSpecGym_retrieval_molecules_4M.tsv"
MOLECULES_1M = "/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/data/massspecgym/MassSpecGym_retrieval_molecules_1M.tsv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def smiles_to_inchi(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToInchi(mol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=str)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = input_path.with_name(input_path.stem + "_canon2.json")

    # --- 1. Load JSON and collect unique candidate SMILES ---
    log.info("Loading %s ...", input_path)
    with open(input_path) as f:
        cands: dict[str, list[str]] = json.load(f)
    log.info("Loaded %d keys", len(cands))

    unique_smiles: set[str] = set()
    for vals in cands.values():
        unique_smiles.update(vals)
    log.info("Unique candidate SMILES: %d", len(unique_smiles))

    # --- 2. Compute InChI and InChIKey for each unique SMILES ---
    log.info("Computing InChI + InChIKey for %d unique SMILES ...", len(unique_smiles))
    smi_list = sorted(unique_smiles)
    smi_to_inchi: dict[str, str] = {}
    smi_to_inchikey: dict[str, str] = {}
    failed_inchi = 0
    for i, smi in enumerate(smi_list):
        if i % 500_000 == 0 and i > 0:
            log.info("  InChI progress: %d / %d", i, len(smi_list))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed_inchi += 1
            continue
        inchi = Chem.MolToInchi(mol)
        if inchi is None:
            failed_inchi += 1
            continue
        smi_to_inchi[smi] = inchi
        smi_to_inchikey[smi] = Chem.InchiToInchiKey(inchi)
    log.info("InChI computed: %d ok, %d failed", len(smi_to_inchi), failed_inchi)

    # --- 3. Build target sets ---
    target_inchis: set[str] = set(smi_to_inchi.values())
    target_inchikeys: set[str] = set(smi_to_inchikey.values())
    log.info("Unique target InChIs: %d, unique target InChIKeys: %d",
             len(target_inchis), len(target_inchikeys))

    # --- 4. Stream PubChem TSV: build InChI and InChIKey maps in one pass ---
    log.info("Streaming %s to build InChI + InChIKey -> PubChem SMILES mappings ...", PUBCHEM_TSV)
    inchi_to_pubchem: dict[str, str] = {}
    inchi_collisions: dict[str, list[str]] = defaultdict(list)
    inchikey_to_pubchem: dict[str, str] = {}
    inchikey_collisions: dict[str, list[str]] = defaultdict(list)

    chunk_iter = pd.read_csv(
        PUBCHEM_TSV, sep="\t", usecols=["smiles", "inchi"],
        dtype=str, chunksize=5_000_000, na_filter=False,
    )
    rows_scanned = 0
    for chunk in chunk_iter:
        # InChI matching
        inchi_mask = chunk["inchi"].isin(target_inchis)
        for _, row in chunk.loc[inchi_mask].iterrows():
            iv, sv = row["inchi"], row["smiles"]
            if iv not in inchi_to_pubchem:
                inchi_to_pubchem[iv] = sv
            else:
                inchi_collisions[iv].append(sv)

        # InChIKey matching (compute key from pubchem inchi for every row in chunk)
        keys = chunk["inchi"].apply(
            lambda x: Chem.InchiToInchiKey(x) if x else None
        )
        ik_mask = keys.isin(target_inchikeys)
        for idx in chunk.index[ik_mask]:
            ik, sv = keys[idx], chunk.at[idx, "smiles"]
            if ik is None:
                continue
            if ik not in inchikey_to_pubchem:
                inchikey_to_pubchem[ik] = sv
            else:
                inchikey_collisions[ik].append(sv)

        rows_scanned += len(chunk)
        if rows_scanned % 25_000_000 == 0:
            log.info("  PubChem rows scanned: %d, InChI matches: %d, InChIKey matches: %d",
                     rows_scanned, len(inchi_to_pubchem), len(inchikey_to_pubchem))

    log.info("PubChem scan done. InChI matches: %d, InChIKey matches: %d",
             len(inchi_to_pubchem), len(inchikey_to_pubchem))

    n_inchi_collisions = sum(1 for v in inchi_collisions.values() if v)
    n_inchikey_collisions = sum(1 for v in inchikey_collisions.values() if v)
    if n_inchi_collisions:
        log.info("InChI collisions (multiple PubChem SMILES): %d", n_inchi_collisions)
        for iv, extras in inchi_collisions.items():
            log.info("  InChI %s... -> kept %s, also saw: %s",
                     iv[:40], inchi_to_pubchem[iv], extras)
    if n_inchikey_collisions:
        log.info("InChIKey collisions (multiple PubChem SMILES): %d", n_inchikey_collisions)
        for ik, extras in inchikey_collisions.items():
            log.info("  InChIKey %s -> kept %s, also saw: %s",
                     ik, inchikey_to_pubchem[ik], extras)

    # --- 5. Build replacement map: InChI first, InChIKey fallback ---
    replacement_map: dict[str, str] = {}
    matched_inchi_different = 0
    matched_inchi_same = 0
    matched_inchikey_different = 0
    matched_inchikey_same = 0
    no_pubchem_match = 0

    for smi in smi_list:
        inchi = smi_to_inchi.get(smi)
        if inchi is None:
            no_pubchem_match += 1
            continue

        # Try InChI match first
        pubchem_smi = inchi_to_pubchem.get(inchi)
        if pubchem_smi is not None:
            if pubchem_smi != smi:
                replacement_map[smi] = pubchem_smi
                matched_inchi_different += 1
            else:
                matched_inchi_same += 1
            continue

        # Fallback: InChIKey match
        ik = smi_to_inchikey.get(smi)
        if ik is None:
            no_pubchem_match += 1
            continue
        pubchem_smi = inchikey_to_pubchem.get(ik)
        if pubchem_smi is not None:
            if pubchem_smi != smi:
                replacement_map[smi] = pubchem_smi
                matched_inchikey_different += 1
            else:
                matched_inchikey_same += 1
        else:
            no_pubchem_match += 1

    log.info("Replacement map built:")
    log.info("  InChI matched (different SMILES): %d", matched_inchi_different)
    log.info("  InChI matched (already identical): %d", matched_inchi_same)
    log.info("  InChIKey matched (different SMILES): %d", matched_inchikey_different)
    log.info("  InChIKey matched (already identical): %d", matched_inchikey_same)
    log.info("  No PubChem match: %d", no_pubchem_match)

    # --- 6. Collect and save unmatched SMILES ---
    unmatched_smiles = set()
    for smi in smi_list:
        if smi in replacement_map:
            continue
        inchi = smi_to_inchi.get(smi)
        if inchi is None:
            unmatched_smiles.add(smi)
            continue
        if inchi in inchi_to_pubchem:
            continue
        ik = smi_to_inchikey.get(smi)
        if ik is not None and ik in inchikey_to_pubchem:
            continue
        unmatched_smiles.add(smi)

    unmatched_tsv_path = input_path.with_name(input_path.stem + "_unmatched_pubchem.tsv")
    log.info("Writing %d unmatched SMILES to %s ...", len(unmatched_smiles), unmatched_tsv_path)
    unmatched_records = [
        {"smiles": smi, "inchi": smi_to_inchi.get(smi, "")}
        for smi in sorted(unmatched_smiles)
    ]
    pd.DataFrame(unmatched_records).to_csv(unmatched_tsv_path, sep="\t", index=False)

    log.info("Checking %d unmatched SMILES against 4M and 1M files ...", len(unmatched_smiles))
    mol4m_smiles = set(pd.read_csv(MOLECULES_4M, sep="\t", usecols=["smiles"])["smiles"])
    mol1m_smiles = set(pd.read_csv(MOLECULES_1M, sep="\t", usecols=["smiles"])["smiles"])

    in_4m = sum(1 for s in unmatched_smiles if s in mol4m_smiles)
    in_1m = sum(1 for s in unmatched_smiles if s in mol1m_smiles)
    in_both = sum(1 for s in unmatched_smiles if s in mol4m_smiles and s in mol1m_smiles)
    in_neither = sum(1 for s in unmatched_smiles if s not in mol4m_smiles and s not in mol1m_smiles)

    log.info("Unmatched SMILES breakdown:")
    log.info("  In 4M file: %d", in_4m)
    log.info("  In 1M file: %d", in_1m)
    log.info("  In both: %d", in_both)
    log.info("  In neither: %d", in_neither)

    # --- 7. Apply replacements and write output ---
    log.info("Applying replacements to JSON ...")
    cands_canon: dict[str, list[str]] = {}
    total_replacements_applied = 0
    for key, vals in cands.items():
        new_vals = []
        for v in vals:
            new_v = replacement_map.get(v, v)
            if new_v != v:
                total_replacements_applied += 1
            new_vals.append(new_v)
        cands_canon[key] = new_vals

    log.info("Total individual replacements applied (incl. duplicates across keys): %d",
             total_replacements_applied)

    log.info("Writing %s ...", output_path)
    with open(output_path, "w") as f:
        json.dump(cands_canon, f)

    # --- 8. Summary ---
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("Input: %s", input_path)
    log.info("Output: %s", output_path)
    log.info("Total keys: %d", len(cands))
    log.info("Unique candidate SMILES: %d", len(unique_smiles))
    log.info("  InChI matched (different SMILES): %d", matched_inchi_different)
    log.info("  InChI matched (already identical): %d", matched_inchi_same)
    log.info("  InChIKey matched (different SMILES): %d", matched_inchikey_different)
    log.info("  InChIKey matched (already identical): %d", matched_inchikey_same)
    log.info("  No PubChem match (after both): %d", no_pubchem_match)
    log.info("  Failed InChI conversion: %d", failed_inchi)
    log.info("InChI collisions in PubChem: %d", n_inchi_collisions)
    log.info("InChIKey collisions in PubChem: %d", n_inchikey_collisions)
    log.info("Unmatched -> in 4M: %d, in 1M: %d, in both: %d, in neither: %d",
             in_4m, in_1m, in_both, in_neither)
    log.info("Total slot-level replacements applied: %d", total_replacements_applied)
    log.info("Done.")


if __name__ == "__main__":
    main()
