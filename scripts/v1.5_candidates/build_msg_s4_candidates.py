#!/usr/bin/env python
"""Build S4-generated formula+mass retrieval candidate JSONs for MassSpecGym.

Pipeline analog of DreaMS-Mol/scripts/data_processing/build_s4_candidates.py
(used for MERLIN_Apr2026), specialized to MassSpecGym v1.5:

- Standardization via ``massspecgym.utils.rdkit_canonical_smiles`` (pure
  RDKit canonical SMILES, matching the v1.5 release).
- Query scope = all folds (train + val + test, ~28,729 unique 2D-InChIKey).
- S4 pretrain is reused from the MERLIN run (ChEMBL31, no leakage). Fine-tune
  is fresh on MSG-train SMILES only (~22,746 unique 2D-InChIKey) — keeps the
  pool free of memorized val/test answers.
- Mass-based candidate buckets: per-molecule neutral exact mass ±10 ppm.
- 1024-cap, uniform-random trim. Query SMILES at index 0.

Stages (toggle via ``--stage``; outputs idempotent so a stage skips if its
artefacts already exist):

  extract       – read data/MassSpecGym1.5.tsv, dedup unique queries by
                  2D-InChIKey across all folds, canonicalize via
                  ``rdkit_canonical_smiles``. Emit extract_unique_mols.parquet
                  with columns (inchikey_2d, smiles, formula, exact_mass, fold).
  prep_train    – train/val split of MSG-train SMILES (95/5, seed=0), packed
                  as zips for s4dd.dataloaders.create_dataloader. Drops mols
                  with OOV tokens vs the pretrain vocab.
  finetune      – continue training MERLIN's ChEMBL31 ckpt_pretrain on
                  MSG-train zips. Per-epoch ckpt + best-by-val-loss finalize.
  sample        – parallel-aware: ``--shard-id N --n-shards M`` so a SLURM
                  array task computes its own slice of (temperature × shard).
                  Output to samples/temp_{T:.2f}/shard_{shard_id:05d}.parquet.
  bucket        – read all sample shards, standardize + compute formula +
                  exact_mass via worker pool, dedup by 2D-InChIKey keeping
                  highest LL. Emit buckets.parquet.
  emit_formula_json – per-query formula bucket → uniform-random trim to 1023
                  + [query]. Write MassSpecGym_S4_retrieval_candidates_formula.json.
  emit_mass_json    – per-query, find pool rows with mass within ±10ppm →
                  trim to 1023 + [query]. Write *mass.json variant.

Designed to run as a SLURM array for the sample stage and as single jobs for
the rest.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

WORKSPACE = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev")
# Read the canon TSV produced by scripts/fixes/rdkit_canon_massspecgym.py
# (stage 0 of the published pipeline). The candidate JSONs are then written
# into the same data/v1.5/ directory.
DEFAULT_TSV = WORKSPACE / "MassSpecGym/data/v1.5/MassSpecGym1.5.tsv"
DEFAULT_OUT_DIR = WORKSPACE / "experiments/data_builds/MassSpecGym_S4"
DEFAULT_FORMULA_JSON = WORKSPACE / "MassSpecGym/data/MassSpecGym_S4_retrieval_candidates_formula_nostereo.json"
DEFAULT_MASS_JSON = WORKSPACE / "MassSpecGym/data/MassSpecGym_S4_retrieval_candidates_mass_nostereo.json"
DEFAULT_PRETRAIN_CKPT = WORKSPACE / "experiments/data_builds/MERLIN_Apr2026/s4/ckpt_pretrain"
DEFAULT_S4_REPO = WORKSPACE / "s4-for-de-novo-drug-design"


# -----------------------------------------------------------------------------
# Lazy imports
# -----------------------------------------------------------------------------

def _import_rdkit():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    return Chem, rdMolDescriptors


def _import_msg_canon():
    """Return a callable ``smi -> canonical_smi``. Uses MSG's pure-RDKit
    canonicalization (``massspecgym.utils.rdkit_canonical_smiles``)."""
    # MSG package is editable-installed under the venv; just import.
    from massspecgym.utils import rdkit_canonical_smiles  # noqa: E402
    return rdkit_canonical_smiles


def _formula_for(smi: str) -> str | None:
    Chem, rdMolDescriptors = _import_rdkit()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        return None


def _inchikey2d_for(smi: str) -> str | None:
    Chem, _ = _import_rdkit()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        ik = Chem.MolToInchiKey(m)
        return ik[:14] if ik else None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Worker pool (canonicalize + formula + ik2d + exact mass)
# -----------------------------------------------------------------------------

def _worker_init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from massspecgym.utils import rdkit_canonical_smiles  # noqa: F401


def _worker_canon(item):
    """Canonicalize a SMILES. Returns (key, canon_smi_or_None)."""
    key, smi = item
    if not isinstance(smi, str) or not smi:
        return key, None
    from massspecgym.utils import rdkit_canonical_smiles
    try:
        out = rdkit_canonical_smiles(smi)
        return key, out
    except Exception:
        return key, None


def _worker_canon_full(item):
    """Canonicalize + compute (ik2d, formula, exact_mass) for the standardized SMILES.

    Returns (key, std_smi, ik2d, formula, exact_mass). Any field may be None
    if the corresponding RDKit call failed.
    """
    key, smi = item
    if not isinstance(smi, str) or not smi:
        return key, None, None, None, None
    from massspecgym.utils import rdkit_canonical_smiles
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    try:
        std = rdkit_canonical_smiles(smi)
    except Exception:
        return key, None, None, None, None
    if not std:
        return key, None, None, None, None
    m = Chem.MolFromSmiles(std)
    if m is None:
        return key, std, None, None, None
    try:
        ik = Chem.MolToInchiKey(m)
        ik2d = ik[:14] if ik else None
    except Exception:
        ik2d = None
    try:
        formula = rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        formula = None
    try:
        exact_mass = float(rdMolDescriptors.CalcExactMolWt(m))
    except Exception:
        exact_mass = None
    return key, std, ik2d, formula, exact_mass


def _run_pool(items, n_workers, worker_fn):
    if n_workers <= 1:
        return [worker_fn(it) for it in items]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        return list(pool.imap_unordered(worker_fn, items, chunksize=512))


# -----------------------------------------------------------------------------
# Stage: extract
# -----------------------------------------------------------------------------

def _stage_extract(args):
    """Load MSG1.5 TSV, canonicalize-and-strip every unique raw SMILES, and emit
    one row per **unique stripped canonical SMILES** (not per stored 2D-InChIKey).

    Keying on the post-strip SMILES guarantees every distinct TSV query molecule
    survives — earlier per-ik2d grouping silently dropped ~25 input ik2ds whose
    RDKit-recomputed 2D-InChIKey collided with another's after canonicalization.
    """
    out_path = args.out_dir / "extract_unique_mols.parquet"
    if out_path.exists() and not args.force:
        print(f"[extract] {out_path} exists; skipping")
        return out_path

    print(f"[extract] reading {args.input_tsv} ...")
    df = pd.read_csv(args.input_tsv, sep="\t", usecols=["smiles", "inchikey", "fold"])
    print(f"[extract] {len(df):,} spectra rows; {df['smiles'].nunique():,} unique raw SMILES")

    # First-fold per RAW SMILES (a mol may appear in only one fold per the disjointness check).
    fold_per_raw = df.groupby("smiles")["fold"].first()
    base = pd.DataFrame({"smiles_raw": fold_per_raw.index, "fold": fold_per_raw.values})
    print(f"[extract] {len(base):,} unique raw SMILES across all folds")

    items = list(zip(base["smiles_raw"], base["smiles_raw"]))
    print(f"[extract] canonicalizing + stripping stereo + computing formula/ik2d/mass ...")
    t0 = time.perf_counter()
    results = _run_pool(items, args.n_workers, _worker_canon_full)
    print(f"[extract]   done in {time.perf_counter() - t0:.1f}s")

    fold_lookup = dict(zip(base["smiles_raw"], base["fold"]))
    rows = []
    n_parse_fail = 0
    for raw_smi, std, ik2d, formula, mass in results:
        if not (std and ik2d and formula and mass is not None):
            n_parse_fail += 1
            continue
        # Strip stereo here so the emit stage's join key is the same form
        # used in MassSpecGym1.5_nostereo.tsv.
        from rdkit import Chem
        m = Chem.MolFromSmiles(std)
        smi_nostereo = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False) if m else std
        rows.append((ik2d, smi_nostereo, formula, mass, fold_lookup[raw_smi]))
    out_df = pd.DataFrame(rows, columns=["inchikey_2d", "smiles", "formula", "exact_mass", "fold"])
    print(f"[extract] {n_parse_fail:,} raw SMILES failed standardization (no formula/mass)")
    # Dedup by the canonical+stripped SMILES (this is what emit will key on).
    # Multiple raw SMILES that strip to the same nostereo form become one entry.
    out_df = out_df.drop_duplicates("smiles", keep="first").reset_index(drop=True)
    print(f"[extract] {len(out_df):,} unique canonical+stripped SMILES")
    print(f"[extract] fold counts:\n{out_df['fold'].value_counts().to_string()}")
    out_df.to_parquet(out_path, compression="zstd")
    print(f"[extract] -> {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Stage: prep_train  (split MSG-train mols into 95/5 train/val zips for S4)
# -----------------------------------------------------------------------------

def _read_smiles_zip(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        with zf.open(zf.namelist()[0]) as g:
            return g.read().decode("utf-8").splitlines()


def _write_smiles_zip(path: Path, smiles_iter):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(smiles_iter)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.stem + ".txt", body)


def _stage_prep_train(args):
    train_zip = args.out_dir / "msg_train.zip"
    val_zip = args.out_dir / "msg_valid.zip"
    if train_zip.exists() and val_zip.exists() and not args.force:
        print(f"[prep_train] {train_zip.name}/{val_zip.name} exist; skipping")
        return train_zip, val_zip

    df = pd.read_parquet(args.out_dir / "extract_unique_mols.parquet")
    train_mols = df.loc[df["fold"] == "train", "smiles"].tolist()
    print(f"[prep_train] {len(train_mols):,} MSG-train unique mols")

    # Filter against pretrain ChEMBL vocab.
    init_args = args.pretrain_ckpt / "init_arguments.json"
    if init_args.exists():
        with init_args.open() as f:
            props = json.load(f)
        vocab = set(props.get("token2label", {}).keys())
        if vocab:
            sys.path.insert(0, str(args.s4_repo))
            from s4dd.smiles_utils import segment_smiles
            keep, drop = [], []
            for s in train_mols:
                if set(segment_smiles(s)) - vocab:
                    drop.append(s)
                else:
                    keep.append(s)
            print(f"[prep_train] dropped {len(drop):,} mols with OOV tokens vs pretrain vocab "
                  f"({len(drop) / max(1, len(train_mols)) * 100:.2f}%)")
            train_mols = keep

    rng = np.random.RandomState(0)
    idx = np.arange(len(train_mols))
    rng.shuffle(idx)
    n_val = max(1, len(train_mols) // 20)
    val_smiles = [train_mols[i] for i in idx[:n_val]]
    train_smiles = [train_mols[i] for i in idx[n_val:]]
    print(f"[prep_train] split — train={len(train_smiles):,} val={len(val_smiles):,}")

    _write_smiles_zip(train_zip, train_smiles)
    _write_smiles_zip(val_zip, val_smiles)
    print(f"[prep_train] -> {train_zip} + {val_zip}")
    return train_zip, val_zip


# -----------------------------------------------------------------------------
# S4 helpers (mirrors DreaMS-Mol/scripts/data_processing/build_s4_candidates.py)
# -----------------------------------------------------------------------------

def _build_s4(args, ckpt_in: Path | None, n_epochs: int | None = None):
    sys.path.insert(0, str(args.s4_repo))
    from s4dd import S4forDenovoDesign
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = args.epochs if n_epochs is None else n_epochs
    print(f"[s4] using device={device}")
    if ckpt_in is not None and ckpt_in.exists():
        print(f"[s4] loading checkpoint dir {ckpt_in}")
        s4 = S4forDenovoDesign.from_file(str(ckpt_in))
        s4.n_max_epochs = epochs
        s4.batch_size = args.batch_size
        s4.device = device
        s4.s4_model.to(device)
        return s4
    s4 = S4forDenovoDesign(n_max_epochs=epochs, batch_size=args.batch_size, device=device)
    return s4


def _save_ckpt(s4, ckpt_out: Path, history: dict | None):
    import json as _json
    ckpt_out.mkdir(parents=True, exist_ok=True)
    s4.save(str(ckpt_out))
    if history is not None:
        with (ckpt_out / "history.json").open("w") as f:
            _json.dump(history, f)
    print(f"[s4] saved {ckpt_out}")


def _epoch_ckpts(epochs_dir: Path):
    if not epochs_dir.exists():
        return []
    out = []
    for p in sorted(epochs_dir.glob("epoch-*")):
        if not p.is_dir():
            continue
        try:
            n = int(p.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if not (p / "model.pt").exists() or not (p / "init_arguments.json").exists():
            continue
        val_loss = float("inf")
        h = p / "history.json"
        if h.exists():
            try:
                with h.open() as f:
                    history = json.load(f)
                vls = history.get("val_loss") or []
                if vls and not (isinstance(vls[-1], float) and vls[-1] != vls[-1]):
                    val_loss = float(vls[-1])
            except Exception:
                pass
        out.append((p, n, val_loss))
    return out


def _latest_epoch_ckpt(epochs_dir: Path):
    cks = _epoch_ckpts(epochs_dir)
    if not cks:
        return None, 0
    p, n, _ = max(cks, key=lambda x: x[1])
    return p, n


def _best_epoch_ckpt(epochs_dir: Path):
    cks = [c for c in _epoch_ckpts(epochs_dir) if c[2] != float("inf")]
    if not cks:
        return None, 0, float("inf")
    p, n, vl = min(cks, key=lambda x: x[2])
    return p, n, vl


class _PerEpochCheckpoint:
    def __init__(self, save_fn, basedir: Path, epoch_offset: int = 0):
        self.save_fn = save_fn
        self.basedir = Path(basedir)
        self.epoch_offset = epoch_offset
        self.stop_training = False

    def on_epoch_end(self, epoch_ix, history, **kwargs):
        import json as _json
        global_ix = self.epoch_offset + epoch_ix + 1
        savedir = self.basedir / f"epoch-{global_ix:03d}"
        savedir.mkdir(parents=True, exist_ok=True)
        self.save_fn(str(savedir))
        with (savedir / "history.json").open("w") as f:
            _json.dump(history, f)
        print(f"[ckpt] saved epoch-{global_ix:03d}", flush=True)

    def on_train_end(self, epoch_ix, history, **kwargs):
        pass


def _stage_finetune(args):
    ckpt_final = args.out_dir / "ckpt_finetune_msg"
    epochs_dir = args.out_dir / "ckpt_finetune_msg_epochs"
    if ckpt_final.exists() and not args.force:
        print(f"[finetune] {ckpt_final} exists; skipping")
        return ckpt_final

    epochs_dir.mkdir(parents=True, exist_ok=True)
    latest_path, latest_n = _latest_epoch_ckpt(epochs_dir)
    if latest_n >= args.epochs and latest_path is not None:
        print(f"[finetune] all {args.epochs} epochs already on disk; finalizing best-by-val-loss")
        bp, bn, bvl = _best_epoch_ckpt(epochs_dir)
        _save_ckpt(_build_s4(args, ckpt_in=bp, n_epochs=0), ckpt_final, history=None)
        return ckpt_final

    if latest_path is not None:
        print(f"[finetune] resuming from {latest_path} (epoch {latest_n}/{args.epochs})")
        remaining = args.epochs - latest_n
        s4 = _build_s4(args, ckpt_in=latest_path, n_epochs=remaining)
        epoch_offset = latest_n
    else:
        s4 = _build_s4(args, ckpt_in=args.pretrain_ckpt, n_epochs=args.epochs)
        epoch_offset = 0

    sys.path.insert(0, str(args.s4_repo))
    from s4dd.torch_callbacks import HistoryLogger
    callbacks = [
        _PerEpochCheckpoint(save_fn=s4.save, basedir=epochs_dir, epoch_offset=epoch_offset),
        HistoryLogger(savedir=str(epochs_dir)),
    ]
    train_zip = args.out_dir / "msg_train.zip"
    val_zip = args.out_dir / "msg_valid.zip"
    print(f"[finetune] train={train_zip}  val={val_zip}  remaining_epochs={s4.n_max_epochs}  bs={args.batch_size}")
    t0 = time.perf_counter()
    s4.train(training_molecules_path=str(train_zip), val_molecules_path=str(val_zip), callbacks=callbacks)
    print(f"[finetune] train() returned after {(time.perf_counter() - t0) / 60:.1f}min")

    bp, bn, bvl = _best_epoch_ckpt(epochs_dir)
    if bp is None:
        raise RuntimeError("[finetune] no per-epoch ckpt with finite val_loss; aborting")
    latest_path, latest_n = _latest_epoch_ckpt(epochs_dir)
    if latest_n != bn:
        print(f"[finetune] WARNING: latest=epoch-{latest_n:03d}, best-by-val-loss=epoch-{bn:03d} "
              f"(val={bvl:.4f}). Promoting best-by-val-loss.")
    print(f"[finetune] finalizing epoch-{bn:03d} (val_loss={bvl:.4f}) -> {ckpt_final}")
    _save_ckpt(_build_s4(args, ckpt_in=bp, n_epochs=0), ckpt_final, history=None)
    return ckpt_final


# -----------------------------------------------------------------------------
# Stage: sample (parallel-aware via --shard-id / --n-shards)
# -----------------------------------------------------------------------------

def _design_molecules_fast(s4, n_designs: int, batch_size: int, temperature: float):
    """GPU-vectorized analog of S4.design_molecules (mirrors MERLIN script)."""
    import torch
    import torch.nn.functional as F
    device = s4.device
    s4.s4_model = s4.s4_model.to(device)
    for module in s4.s4_model.modules():
        if hasattr(module, "setup_step"):
            module.setup_step()
    s4.s4_model.eval()

    beg_label = s4.token2label["[BEG]"]
    label2token = s4.label2token
    seq_len = s4.sequence_length

    n_batches = math.ceil(n_designs / batch_size)
    out_smiles, out_mean_lls = [], []
    with torch.no_grad():
        for batch_idx in range(n_batches):
            this_bs = batch_size if batch_idx < n_batches - 1 else n_designs - batch_idx * batch_size
            X = torch.full((this_bs,), beg_label, dtype=torch.int64, device=device)
            s4.s4_model.reset_state(this_bs, device=device)
            sampled_steps, ll_steps = [], []
            for _ in range(seq_len):
                preds = s4.s4_model.recurrent_step(X)
                softmax_preds = F.softmax(preds, dim=-1)
                probs_t = F.softmax(preds / temperature, dim=-1)
                sampled = torch.multinomial(probs_t, num_samples=1).squeeze(-1)
                ll = torch.gather(softmax_preds, 1, sampled.unsqueeze(-1)).squeeze(-1)
                sampled_steps.append(sampled)
                ll_steps.append(ll)
                X = sampled
            designs = torch.stack(sampled_steps, dim=1).cpu().numpy()
            lls = torch.stack(ll_steps, dim=1).cpu().numpy()
            log_lls = np.log(np.clip(lls, 1e-12, 1.0))
            for i in range(this_bs):
                row = designs[i]
                tokens = []
                for lab in row:
                    tok = label2token[int(lab)]
                    if tok in ("[BEG]", "[END]", "[PAD]"):
                        continue
                    tokens.append(tok)
                mol_len = len(tokens) + 2
                cutoff = min(seq_len, mol_len - 1)
                mean_ll = float(np.mean(log_lls[i, :cutoff])) if cutoff > 0 else 0.0
                out_smiles.append("".join(tokens))
                out_mean_lls.append(mean_ll)
    return out_smiles, out_mean_lls


def _stage_sample(args):
    """Each invocation handles ONE task-id slice across all temperatures. Each
    ``chunk_size`` block is written to its own parquet shard so progress
    survives crashes — shards are named ``shard_{task_id:03d}_{chunk:05d}.parquet``."""
    src_ckpt = args.out_dir / "ckpt_finetune_msg"
    samples_dir = args.out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    s4 = _build_s4(args, ckpt_in=src_ckpt)

    temperatures = [float(t) for t in args.temperatures.split(",")]
    n_per_temp = args.n_designs // max(1, len(temperatures))
    n_shards = max(1, args.n_shards)
    n_per_task_per_temp = math.ceil(n_per_temp / n_shards)
    print(f"[sample] total={args.n_designs:,} temps={temperatures} per_temp={n_per_temp:,} "
          f"n_shards={n_shards} per_task_per_temp={n_per_task_per_temp:,} chunk={args.chunk_size}")

    schema = pa.schema([
        ("smiles", pa.string()),
        ("log_likelihood", pa.float32()),
        ("temperature", pa.float32()),
    ])

    task_id = args.shard_id  # 0..n_shards-1

    for temp in temperatures:
        t_dir = samples_dir / f"temp_{temp:.2f}"
        t_dir.mkdir(parents=True, exist_ok=True)
        # Count how many of our own chunks are already on disk.
        existing = sorted(t_dir.glob(f"shard_{task_id:03d}_*.parquet"))
        n_done = sum(pq.read_metadata(p).num_rows for p in existing)
        if n_done >= n_per_task_per_temp:
            print(f"[sample] T={temp} task {task_id}: {n_done:,} >= {n_per_task_per_temp:,} already; skipping")
            continue
        next_chunk = len(existing)
        n_remaining = n_per_task_per_temp - n_done
        print(f"[sample] T={temp} task {task_id}: {n_done:,} on disk, {n_remaining:,} remaining, "
              f"next chunk idx={next_chunk}")

        while n_remaining > 0:
            this_chunk = min(args.chunk_size, n_remaining)
            t_chunk = time.perf_counter()
            designs, lls = _design_molecules_fast(s4, n_designs=this_chunk, batch_size=args.batch_size, temperature=temp)
            elapsed = time.perf_counter() - t_chunk
            shard_path = t_dir / f"shard_{task_id:03d}_{next_chunk:05d}.parquet"
            df = pd.DataFrame({
                "smiles": designs,
                "log_likelihood": np.asarray(lls, dtype=np.float32),
                "temperature": np.full(len(designs), temp, dtype=np.float32),
            })
            table = pa.Table.from_pandas(df, preserve_index=False, schema=schema)
            pq.write_table(table, shard_path, compression="zstd")
            print(f"[sample] T={temp} task {task_id} chunk {next_chunk}: {len(designs):,} smiles in {elapsed:.1f}s "
                  f"({len(designs)/elapsed:.1f}/s) -> {shard_path.name}", flush=True)
            n_remaining -= len(designs)
            next_chunk += 1


# -----------------------------------------------------------------------------
# Stage: bucket
# -----------------------------------------------------------------------------

def _stage_bucket(args):
    bucket_path = args.out_dir / "buckets.parquet"
    if bucket_path.exists() and not args.force:
        print(f"[bucket] {bucket_path} exists; skipping")
        return bucket_path

    samples_dir = args.out_dir / "samples"
    shard_paths = sorted(samples_dir.glob("temp_*/shard_*.parquet"))
    print(f"[bucket] {len(shard_paths):,} sample shards")

    seen: dict[str, tuple[float, float]] = {}  # raw_smi -> (best_ll, temp)
    n_raw = 0
    for sp in shard_paths:
        df = pd.read_parquet(sp, columns=["smiles", "log_likelihood", "temperature"])
        n_raw += len(df)
        for smi, ll, t in df.itertuples(index=False, name=None):
            cur = seen.get(smi)
            if cur is None or ll > cur[0]:
                seen[smi] = (float(ll), float(t))
    print(f"[bucket] raw rows={n_raw:,}  unique raw smiles={len(seen):,}")

    items = [(smi, smi) for smi in seen]
    print(f"[bucket] canonicalizing + computing formula/ik2d/mass via worker pool ({args.n_workers}) ...")
    t0 = time.perf_counter()
    results = _run_pool(items, args.n_workers, _worker_canon_full)
    print(f"[bucket]   done in {time.perf_counter() - t0:.1f}s")

    rows = []
    for raw_smi, std_smi, ik2d, formula, mass in results:
        if not (std_smi and ik2d and formula and mass is not None):
            continue
        ll, temp = seen[raw_smi]
        rows.append((std_smi, ik2d, formula, mass, ll, temp))
    df_pool = pd.DataFrame(rows, columns=["smiles", "inchikey_2d", "formula", "exact_mass", "log_likelihood", "temperature"])
    print(f"[bucket] survived standardization: {len(df_pool):,}")

    df_pool = (df_pool.sort_values("log_likelihood", ascending=False)
                       .drop_duplicates("inchikey_2d", keep="first")
                       .reset_index(drop=True))
    print(f"[bucket] after dedup by 2D-InChIKey: {len(df_pool):,}")
    df_pool.to_parquet(bucket_path, compression="zstd")
    print(f"[bucket] -> {bucket_path}")
    return bucket_path


# -----------------------------------------------------------------------------
# Stage: emit_formula_json + emit_mass_json
# -----------------------------------------------------------------------------

MAX_CANDIDATES = 512
PPM_MASS = 10  # ±10 ppm tolerance for mass-based buckets


def _trim_uniform(rng: random.Random, candidates: list[str], cap: int) -> list[str]:
    if len(candidates) <= cap:
        return candidates
    return rng.sample(candidates, cap)


def _build_stereo_strip_map(unique_smiles: set[str], n_workers: int) -> dict[str, str | None]:
    """Stereo-strip a set of SMILES in parallel. Returns ``{raw_smi -> stripped_or_None}``."""
    from massspecgym.utils import _strip_stereo_parallel
    print(f"[emit] stereo-stripping {len(unique_smiles):,} unique SMILES with {n_workers} workers ...")
    t0 = time.perf_counter()
    out = _strip_stereo_parallel(unique_smiles, n_workers=n_workers)
    print(f"[emit] done in {time.perf_counter()-t0:.1f}s; "
          f"failed={sum(1 for v in out.values() if not v):,}")
    return out


def _stage_emit_formula_json(args):
    queries = pd.read_parquet(args.out_dir / "extract_unique_mols.parquet")
    pool = pd.read_parquet(args.out_dir / "buckets.parquet")
    print(f"[emit_formula] queries={len(queries):,}  pool={len(pool):,}")

    # Stereo-strip query + pool SMILES before forming candidate lists. 2D-InChIKey
    # is stereo-invariant so the existing ik2d columns remain valid. Per-query
    # dedup uses ik2d (not SMILES) so candidates with the same 2D structure as the
    # query are excluded even if their canonical SMILES differs (e.g. tautomers).
    smi2no = _build_stereo_strip_map(set(queries["smiles"]).union(pool["smiles"]), args.n_workers)
    queries = queries.assign(smiles_no=queries["smiles"].map(smi2no)).dropna(subset=["smiles_no"]).reset_index(drop=True)
    pool = pool.assign(smiles_no=pool["smiles"].map(smi2no)).dropna(subset=["smiles_no"]).reset_index(drop=True)
    print(f"[emit_formula] post-strip — queries={len(queries):,}  pool={len(pool):,}")

    formula_to_entries: dict[str, list[tuple[str, str]]] = {}
    for row in pool[["formula", "smiles_no", "inchikey_2d"]].itertuples(index=False):
        formula_to_entries.setdefault(row.formula, []).append((row.smiles_no, row.inchikey_2d))
    print(f"[emit_formula] {len(formula_to_entries):,} unique formulas in pool")

    rng = random.Random(0)
    out: dict[str, list[str]] = {}
    sizes = []
    n_trivial = 0
    for q_smi, q_formula, q_ik in zip(queries["smiles_no"], queries["formula"], queries["inchikey_2d"]):
        bucket = [smi for (smi, ik) in formula_to_entries.get(q_formula, []) if ik != q_ik]
        if bucket:
            cands = _trim_uniform(rng, bucket, MAX_CANDIDATES - 1)
            out[q_smi] = [q_smi] + cands
        else:
            out[q_smi] = [q_smi]
            n_trivial += 1
        sizes.append(len(out[q_smi]))
    print(f"[emit_formula] trivial fallback: {n_trivial:,} ({n_trivial/len(queries)*100:.2f}%)")
    print(f"[emit_formula] list size — median={int(np.median(sizes))} mean={np.mean(sizes):.1f} "
          f"p90={int(np.percentile(sizes, 90))} max={int(max(sizes))}")
    args.out_formula_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_formula_json.open("w") as f:
        json.dump(out, f)
    print(f"[emit_formula] wrote {args.out_formula_json} ({os.path.getsize(args.out_formula_json)/1e6:.1f} MB)")


def _stage_emit_mass_json(args):
    queries = pd.read_parquet(args.out_dir / "extract_unique_mols.parquet")
    pool = pd.read_parquet(args.out_dir / "buckets.parquet")
    print(f"[emit_mass] queries={len(queries):,}  pool={len(pool):,}")

    smi2no = _build_stereo_strip_map(set(queries["smiles"]).union(pool["smiles"]), args.n_workers)
    queries = queries.assign(smiles_no=queries["smiles"].map(smi2no)).dropna(subset=["smiles_no"]).reset_index(drop=True)
    pool = pool.assign(smiles_no=pool["smiles"].map(smi2no)).dropna(subset=["smiles_no"]).reset_index(drop=True)
    print(f"[emit_mass] post-strip — queries={len(queries):,}  pool={len(pool):,}")

    pool_sorted = pool.sort_values("exact_mass").reset_index(drop=True)
    masses = pool_sorted["exact_mass"].to_numpy()
    smiles_no = pool_sorted["smiles_no"].to_numpy()
    iks = pool_sorted["inchikey_2d"].to_numpy()

    rng = random.Random(0)
    out: dict[str, list[str]] = {}
    sizes = []
    n_trivial = 0
    for q_smi, q_mass, q_ik in zip(queries["smiles_no"], queries["exact_mass"], queries["inchikey_2d"]):
        tol = q_mass * PPM_MASS * 1e-6
        lo = np.searchsorted(masses, q_mass - tol, side="left")
        hi = np.searchsorted(masses, q_mass + tol, side="right")
        slc_smi = smiles_no[lo:hi]
        slc_ik = iks[lo:hi]
        bucket = [slc_smi[i] for i in range(len(slc_smi)) if slc_ik[i] != q_ik]
        if bucket:
            cands = _trim_uniform(rng, bucket, MAX_CANDIDATES - 1)
            out[q_smi] = [q_smi] + cands
        else:
            out[q_smi] = [q_smi]
            n_trivial += 1
        sizes.append(len(out[q_smi]))
    print(f"[emit_mass] trivial fallback: {n_trivial:,} ({n_trivial/len(queries)*100:.2f}%)")
    print(f"[emit_mass] list size — median={int(np.median(sizes))} mean={np.mean(sizes):.1f} "
          f"p90={int(np.percentile(sizes, 90))} max={int(max(sizes))}")
    args.out_mass_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_mass_json.open("w") as f:
        json.dump(out, f)
    print(f"[emit_mass] wrote {args.out_mass_json} ({os.path.getsize(args.out_mass_json)/1e6:.1f} MB)")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

STAGES = ["extract", "prep_train", "finetune", "sample", "bucket", "emit_formula_json", "emit_mass_json"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=STAGES + ["all"])
    p.add_argument("--input-tsv", type=Path, default=DEFAULT_TSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--out-formula-json", type=Path, default=DEFAULT_FORMULA_JSON)
    p.add_argument("--out-mass-json", type=Path, default=DEFAULT_MASS_JSON)
    p.add_argument("--pretrain-ckpt", type=Path, default=DEFAULT_PRETRAIN_CKPT)
    p.add_argument("--s4-repo", type=Path, default=DEFAULT_S4_REPO)

    # S4 hyperparams.
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)

    # Sample.
    p.add_argument("--n-designs", type=int, default=200_000_000)
    p.add_argument("--temperatures", type=str, default="0.7,1.0,1.2")
    p.add_argument("--chunk-size", type=int, default=200_000)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)

    p.add_argument("--n-workers", type=int, default=int(os.environ.get("MSG_S4_WORKERS", 32)))
    p.add_argument("--force", action="store_true")

    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build_msg_s4_candidates] stage={args.stage} out_dir={args.out_dir}")

    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n=== STAGE: {stage} ===")
        fn = globals()[f"_stage_{stage}"]
        fn(args)


if __name__ == "__main__":
    main()
