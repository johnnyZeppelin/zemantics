from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from dataset import WikiLinguaGroupDataset
from model import LatentRendererModel


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[LatentRendererModel, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    backbone = cfg.get("backbone", "google/mt5-small")
    num_latents = int(cfg.get("num_latents", 16))
    latent_dropout = float(cfg.get("latent_dropout", 0.1))

    model = LatentRendererModel(
        backbone_name=backbone,
        num_latents=num_latents,
        latent_dropout=latent_dropout,
        latent_noise_std=0.0,  # probe时关掉噪声
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def tokenize_docs(tokenizer, texts: List[str], max_len: int) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def length_bucket(lengths: torch.Tensor) -> torch.Tensor:
    """
    lengths: [B] integer token lengths
    buckets:
      0: 1-32
      1: 33-64
      2: 65-128
      3: 129+
    """
    b = torch.zeros_like(lengths)
    b = b + (lengths > 32).long()
    b = b + (lengths > 64).long()
    b = b + (lengths > 128).long()
    return b.clamp(0, 3)


@torch.no_grad()
def extract_features(
    model: LatentRendererModel,
    tokenizer,
    jsonl_path: str,
    device: torch.device,
    max_groups: Optional[int],
    max_doc_len: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each group:
      - compute zbar for en_doc and zh_doc
      - create two samples:
          x = zbar
          y_lang = 0 for en, 1 for zh
          y_len  = length bucket (based on token length)
    Returns:
      X: [2N, D]
      y_lang: [2N]
      y_len: [2N]
    """
    ds = WikiLinguaGroupDataset(jsonl_path, max_examples=max_groups)

    # We do lightweight batching ourselves for speed
    X_list: List[torch.Tensor] = []
    y_lang_list: List[torch.Tensor] = []
    y_len_list: List[torch.Tensor] = []

    d_model = model.config.d_model

    for i0 in range(0, len(ds), batch_size):
        batch = ds.data[i0 : i0 + batch_size]
        en_docs = [ex.en_doc for ex in batch]
        zh_docs = [ex.zh_doc for ex in batch]

        en_tok = tokenize_docs(tokenizer, en_docs, max_len=max_doc_len)
        zh_tok = tokenize_docs(tokenizer, zh_docs, max_len=max_doc_len)

        en_ids = en_tok["input_ids"].to(device)
        en_m = en_tok["attention_mask"].to(device)
        zh_ids = zh_tok["input_ids"].to(device)
        zh_m = zh_tok["attention_mask"].to(device)

        # token lengths
        en_len = en_m.sum(dim=1).long().cpu()
        zh_len = zh_m.sum(dim=1).long().cpu()
        en_len_bucket = length_bucket(en_len)
        zh_len_bucket = length_bucket(zh_len)

        # zbar
        z_en = model.encode_to_z(en_ids, en_m, add_noise=False)
        z_zh = model.encode_to_z(zh_ids, zh_m, add_noise=False)

        zbar_en = model.pool_zbar(z_en).detach().cpu()  # [B, D]
        zbar_zh = model.pool_zbar(z_zh).detach().cpu()

        # stack two views as two samples
        X = torch.cat([zbar_en, zbar_zh], dim=0)  # [2B, D]
        y_lang = torch.cat(
            [torch.zeros(zbar_en.size(0), dtype=torch.long), torch.ones(zbar_zh.size(0), dtype=torch.long)],
            dim=0,
        )  # [2B]
        y_len = torch.cat([en_len_bucket, zh_len_bucket], dim=0)  # [2B]

        assert X.size(1) == d_model

        X_list.append(X)
        y_lang_list.append(y_lang)
        y_len_list.append(y_len)

    X_all = torch.cat(X_list, dim=0)
    y_lang_all = torch.cat(y_lang_list, dim=0)
    y_len_all = torch.cat(y_len_list, dim=0)
    return X_all, y_lang_all, y_len_all


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 0) -> None:
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_valid: torch.Tensor,
    y_valid: torch.Tensor,
    out_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden: int = 0,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_train.size(1)

    probe = LinearProbe(in_dim, out_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    def acc_on(dl) -> float:
        probe.eval()
        total = 0
        hit = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = probe(xb)
            pred = logits.argmax(dim=1)
            hit += (pred == yb).sum().item()
            total += yb.numel()
        return hit / max(total, 1)

    best = 0.0
    for ep in range(1, epochs + 1):
        probe.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        tr_acc = acc_on(train_dl)
        va_acc = acc_on(valid_dl)
        best = max(best, va_acc)

        if ep in [1, 2, 5, 10, epochs]:
            print({"epoch": ep, "train_acc": tr_acc, "valid_acc": va_acc})

    return {"train_acc": float(tr_acc), "valid_acc": float(va_acc), "best_valid_acc": float(best)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
    ap.add_argument("--max_train_groups", type=int, default=2000)
    ap.add_argument("--max_valid_groups", type=int, default=500)
    ap.add_argument("--feat_batch_size", type=int, default=16)

    ap.add_argument("--probe_hidden", type=int, default=0, help="0 means linear probe")
    ap.add_argument("--probe_batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-2)

    ap.add_argument("--out", default="", help="default: <run_dir>/probe_leakage.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_from_ckpt(args.ckpt, device=device)

    backbone = cfg.get("backbone", "google/mt5-small")
    tok = AutoTokenizer.from_pretrained(backbone)

    max_doc_len = int(cfg.get("max_doc_len", 256))

    print("Extracting features (train)...")
    Xtr, ytr_lang, ytr_len = extract_features(
        model=model,
        tokenizer=tok,
        jsonl_path=args.train_jsonl,
        device=device,
        max_groups=(args.max_train_groups or None),
        max_doc_len=max_doc_len,
        batch_size=args.feat_batch_size,
    )
    print("Extracting features (valid)...")
    Xva, yva_lang, yva_len = extract_features(
        model=model,
        tokenizer=tok,
        jsonl_path=args.valid_jsonl,
        device=device,
        max_groups=(args.max_valid_groups or None),
        max_doc_len=max_doc_len,
        batch_size=args.feat_batch_size,
    )

    # Move features to CPU for probe training (probe will move batches to device)
    Xtr = Xtr.float().cpu()
    Xva = Xva.float().cpu()

    print(f"Feature shapes: Xtr={tuple(Xtr.shape)}, Xva={tuple(Xva.shape)}")

    # Baselines
    lang_major = max((ytr_lang == 0).float().mean().item(), (ytr_lang == 1).float().mean().item())
    len_counts = torch.bincount(ytr_len, minlength=4).float()
    len_major = (len_counts.max() / len_counts.sum()).item()

    print({"baseline_majority_lang_acc": lang_major, "baseline_majority_len_acc": len_major})

    print("\n=== Probe: Language ID (en vs zh) ===")
    lang_res = train_probe(
        X_train=Xtr,
        y_train=ytr_lang,
        X_valid=Xva,
        y_valid=yva_lang,
        out_dim=2,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.probe_batch_size,
        hidden=args.probe_hidden,
    )

    print("\n=== Probe: Length bucket (4-way) ===")
    len_res = train_probe(
        X_train=Xtr,
        y_train=ytr_len,
        X_valid=Xva,
        y_valid=yva_len,
        out_dim=4,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.probe_batch_size,
        hidden=args.probe_hidden,
    )

    run_dir = os.path.dirname(args.ckpt)
    out_path = args.out or os.path.join(run_dir, "probe_leakage.json")
    out = {
        "ckpt": args.ckpt,
        "max_train_groups": args.max_train_groups,
        "max_valid_groups": args.max_valid_groups,
        "probe_hidden": args.probe_hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "baseline_majority_lang_acc": lang_major,
        "baseline_majority_len_acc": len_major,
        "lang_probe": lang_res,
        "len_probe": len_res,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== SAVED ===")
    print(json.dumps(out, indent=2))
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
