"""ChemBERTa binary-classifier baseline for the MassSpecGym formula challenge.

Trains a ChemBERTa-77M-MLM head to predict whether a single SMILES is a query
(positive) or a candidate/decoy (negative) -- ignoring spectra entirely. At
test time each candidate is scored independently and ranked by P(class=1).

Inputs (paths relative to MassSpecGym/):
- data/MassSpecGym.tsv (NOT the RDKit-canonicalized variant)
- data/MassSpecGym_retrieval_candidates_formula.json (NOT the _RDKit_SMILES variant)

Output:
- data/test_results_v1.5/retrieval/chemberta_binary_test_formula.pkl
- scripts/chemberta_binary_ckpt/ (HuggingFace model checkpoint)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# -- Paths (defaults; overridable via CLI args) -------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
DEFAULT_TSV_PTH = DATA / "MassSpecGym.tsv"
DEFAULT_CANDS_PTH = DATA / "MassSpecGym_retrieval_candidates_formula.json"
DEFAULT_CKPT_DIR = REPO_ROOT / "scripts" / "chemberta_binary_ckpt"
DEFAULT_OUT_PKL = DATA / "test_results_v1.5" / "retrieval" / "chemberta_binary_test_formula.pkl"

# -- Defaults -----------------------------------------------------------------
MODEL_ID = "DeepChem/ChemBERTa-77M-MLM"
MAX_LEN = 128
BATCH_SIZE = 64
LR = 5e-5
WEIGHT_DECAY = 0.01
EPOCHS = 2
INFER_BATCH = 256
SEED = 0


# -- Dataset ------------------------------------------------------------------
class SmilesBinaryDataset(Dataset):
    def __init__(self, smiles: list[str], labels: list[int], tokenizer, max_len: int):
        assert len(smiles) == len(labels)
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_train_pairs(tsv: pd.DataFrame, cands: dict[str, list[str]], seed: int = SEED):
    """For each train query: emit (q, 1) and one randomly-chosen (decoy, 0)."""
    rng = random.Random(seed)
    train_q = sorted(tsv[tsv["fold"] == "train"]["smiles"].dropna().unique().tolist())
    smiles, labels = [], []
    skipped = 0
    for q in train_q:
        if q not in cands:
            skipped += 1
            continue
        cand_list = cands[q]
        decoys = [c for c in cand_list[1:] if c != q]
        if not decoys:
            skipped += 1
            continue
        smiles.append(q)
        labels.append(1)
        smiles.append(rng.choice(decoys))
        labels.append(0)
    return smiles, labels, len(train_q), skipped


# -- Train --------------------------------------------------------------------
def train(model, tokenizer, smiles, labels, device, epochs: int, batch_size: int,
          lr: float, weight_decay: float):
    ds = SmilesBinaryDataset(smiles, labels, tokenizer, MAX_LEN)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)
    total_steps = len(dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(100, total_steps // 20),
        num_training_steps=total_steps,
    )

    model.train()
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        running = 0.0
        running_n = 0
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.amp.autocast(device_type="cpu", enabled=False)
            )
            with ctx:
                out = model(**batch)
                loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * batch["input_ids"].size(0)
            running_n += batch["input_ids"].size(0)
            step += 1
            if step % 200 == 0:
                elapsed = time.time() - t0
                print(
                    f"  step {step}/{total_steps} | ep {ep} | "
                    f"loss {running / running_n:.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed/60:.1f}m elapsed",
                    flush=True,
                )
        print(
            f"epoch {ep} done | mean loss {running / running_n:.4f} | "
            f"{(time.time() - t0)/60:.1f}m elapsed",
            flush=True,
        )


# -- Inference ----------------------------------------------------------------
@torch.no_grad()
def score_smiles(model, tokenizer, smiles_list: list[str], device, batch_size: int):
    model.eval()
    scores = np.zeros(len(smiles_list), dtype=np.float32)
    for i in range(0, len(smiles_list), batch_size):
        chunk = smiles_list[i : i + batch_size]
        enc = tokenizer(
            chunk,
            max_length=MAX_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.amp.autocast(device_type="cpu", enabled=False)
        )
        with ctx:
            logits = model(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1)[:, 1].detach().cpu().numpy()
        scores[i : i + batch_size] = probs
    return scores


def evaluate(model, tokenizer, tsv: pd.DataFrame, cands: dict, device, infer_batch: int,
             smoke: int = 0, incremental_pkl: T.Optional["Path"] = None,
             incremental_every: int = 1000):
    """For each test query, score its candidates, sort, and compute metrics."""
    from massspecgym.utils import MyopicMCES

    mces = MyopicMCES()

    test_rows = tsv[tsv["fold"] == "test"].copy()
    if smoke:
        test_rows = test_rows.head(smoke)
    print(f"evaluating {len(test_rows)} test queries", flush=True)

    rows = {
        "identifier": [],
        "sorted_scores": [],
        "sorted_candidate_smiles": [],
        "test_hit_rate@1": [],
        "test_hit_rate@5": [],
        "test_hit_rate@20": [],
        "test_mrr": [],
        "test_mces@1": [],
    }
    skipped = 0
    t0 = time.time()
    for ri, r in enumerate(test_rows.itertuples(index=False)):
        q = r.smiles
        ident = r.identifier
        if q not in cands:
            skipped += 1
            continue
        cand_list = cands[q]
        if not cand_list:
            skipped += 1
            continue
        scores = score_smiles(model, tokenizer, cand_list, device, infer_batch)
        order = np.argsort(-scores)
        sorted_scores = scores[order].tolist()
        sorted_smis = [cand_list[j] for j in order]
        # GT is index 0 in the candidate list (verified by candidate-pool
        # construction); rank of the GT in the sorted list:
        gt_rank = int(np.where(order == 0)[0][0])
        rows["identifier"].append(ident)
        rows["sorted_scores"].append(sorted_scores)
        rows["sorted_candidate_smiles"].append(sorted_smis)
        rows["test_hit_rate@1"].append(float(gt_rank < 1))
        rows["test_hit_rate@5"].append(float(gt_rank < 5))
        rows["test_hit_rate@20"].append(float(gt_rank < 20))
        rows["test_mrr"].append(1.0 / (gt_rank + 1))
        rows["test_mces@1"].append(float(mces(q, sorted_smis[0])))
        if (ri + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ri + 1) * (len(test_rows) - ri - 1)
            print(
                f"  evaluated {ri + 1}/{len(test_rows)} | "
                f"hit@1 so far {np.mean(rows['test_hit_rate@1'])*100:.2f}% | "
                f"{elapsed/60:.1f}m elapsed | ETA {eta/60:.1f}m",
                flush=True,
            )
        if incremental_pkl is not None and (ri + 1) % incremental_every == 0:
            try:
                pd.DataFrame(rows).to_pickle(str(incremental_pkl) + ".partial")
            except Exception as e:
                print(f"  WARN: incremental save failed: {e}", flush=True)
    if skipped:
        print(f"  skipped {skipped} queries with missing/empty candidate lists", flush=True)
    return pd.DataFrame(rows)


# -- Main ---------------------------------------------------------------------
def main(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tsv_pth = Path(args.tsv_pth)
    cands_pth = Path(args.cands_pth)
    out_pkl = Path(args.out_pkl)
    ckpt_dir = Path(args.ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"  tsv:   {tsv_pth}", flush=True)
    print(f"  cands: {cands_pth}", flush=True)
    print(f"  out:   {out_pkl}", flush=True)
    print(f"  ckpt:  {ckpt_dir}", flush=True)

    print(f"loading TSV {tsv_pth}", flush=True)
    tsv = pd.read_csv(tsv_pth, sep="\t", usecols=["identifier", "smiles", "fold"])

    print(f"loading candidates {cands_pth}", flush=True)
    t0 = time.time()
    with open(cands_pth) as f:
        cands = json.load(f)
    print(f"  loaded {len(cands)} candidate sets in {time.time()-t0:.0f}s", flush=True)

    print(f"loading tokenizer / model {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    model.to(device)

    if not args.skip_train:
        if args.load_ckpt and ckpt_dir.exists():
            print(f"loading existing ckpt {ckpt_dir}", flush=True)
            model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        else:
            print("building training pairs", flush=True)
            t0 = time.time()
            smiles, labels, n_train_q, skipped = build_train_pairs(tsv, cands, seed=SEED)
            if args.smoke:
                smiles = smiles[: args.smoke * 2]
                labels = labels[: args.smoke * 2]
            print(
                f"  {len(smiles)} training pairs from {n_train_q} train queries "
                f"({skipped} queries skipped, no decoys) "
                f"in {time.time()-t0:.0f}s",
                flush=True,
            )
            print(
                f"training: epochs={args.epochs}, batch={args.batch_size}, lr={LR}",
                flush=True,
            )
            train(
                model,
                tokenizer,
                smiles,
                labels,
                device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"saved checkpoint to {ckpt_dir}", flush=True)

    print("evaluating on test fold", flush=True)
    df = evaluate(model, tokenizer, tsv, cands, device, INFER_BATCH, smoke=args.smoke,
                  incremental_pkl=out_pkl, incremental_every=500)
    print(
        f"final test hit@1 = {df['test_hit_rate@1'].mean()*100:.3f}% "
        f"hit@5 = {df['test_hit_rate@5'].mean()*100:.3f}% "
        f"hit@20 = {df['test_hit_rate@20'].mean()*100:.3f}% "
        f"mrr = {df['test_mrr'].mean()*100:.3f}%",
        flush=True,
    )
    if not args.smoke:
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(out_pkl)
        print(f"wrote {out_pkl}", flush=True)
    else:
        smoke_pth = out_pkl.with_name(out_pkl.stem + "_SMOKE.pkl")
        df.to_pickle(smoke_pth)
        print(f"[smoke] wrote {smoke_pth}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tsv_pth", type=str, default=str(DEFAULT_TSV_PTH),
                   help="Path to MassSpecGym TSV (with identifier/smiles/fold columns)")
    p.add_argument("--cands_pth", type=str, default=str(DEFAULT_CANDS_PTH),
                   help="Path to candidates JSON (keyed by query SMILES)")
    p.add_argument("--out_pkl", type=str, default=str(DEFAULT_OUT_PKL),
                   help="Path for the output pickle")
    p.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CKPT_DIR),
                   help="Directory to write/read the HF checkpoint")
    p.add_argument("--smoke", type=int, default=0,
                   help="If >0, use this many train queries and test queries (smoke test)")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--load_ckpt", action="store_true",
                   help="Reuse existing ckpt_dir if present (skip training)")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training entirely (assume model is already loaded)")
    args = p.parse_args()
    main(args)
