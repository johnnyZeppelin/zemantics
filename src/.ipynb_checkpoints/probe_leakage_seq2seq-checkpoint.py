from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from dataset import WikiLinguaGroupDataset


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    backbone = cfg.get("backbone", "google/mt5-small")

    model = AutoModelForSeq2SeqLM.from_pretrained(backbone)
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
    b = torch.zeros_like(lengths)
    b = b + (lengths > 32).long()
    b = b + (lengths > 64).long()
    b = b + (lengths > 128).long()
    return b.clamp(0, 3)


def mean_pool_hidden(h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    m = attn_mask.unsqueeze(-1).float()
    denom = m.sum(dim=1).clamp(min=1.0)
    pooled = (h * m).sum(dim=1) / denom
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled


@torch.no_grad()
def extract_features(
    model,
    tokenizer,
    jsonl_path: str,
    device: torch.device,
    max_groups: Optional[int],
    max_doc_len: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ds = WikiLinguaGroupDataset(jsonl_path, max_examples=max_groups)

    X_list: List[torch.Tensor] = []
    y_lang_list: List[torch.Tensor] = []
    y_len_list: List[torch.Tensor] = []

    encoder = model.get_encoder()

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

        en_len = en_m.sum(dim=1).long().cpu()
        zh_len = zh_m.sum(dim=1).long().cpu()
        en_len_bucket = length_bucket(en_len)
        zh_len_bucket = length_bucket(zh_len)

        h_en = encoder(input_ids=en_ids, attention_mask=en_m, return_dict=True).last_hidden_state
        h_zh = encoder(input_ids=zh_ids, attention_mask=zh_m, return_dict=True).last_hidden_state

        zbar_en = mean_pool_hidden(h_en, en_m).detach().cpu()
        zbar_zh = mean_pool_hidden(h_zh, zh_m).detach().cpu()

        X = torch.cat([zbar_en, zbar_zh], dim=0)
        y_lang = torch.cat(
            [torch.zeros(zbar_en.size(0), dtype=torch.long), torch.ones(zbar_zh.size(0), dtype=torch.long)],
            dim=0,
        )
        y_len = torch.cat([en_len_bucket, zh_len_bucket], dim=0)

        X_list.append(X)
        y_lang_list.append(y_lang)
        y_len_list.append(y_len)

    X_all = torch.cat(X_list, dim=0).float().cpu()
    y_lang_all = torch.cat(y_lang_list, dim=0).cpu()
    y_len_all = torch.cat(y_len_list, dim=0).cpu()
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
            pred = probe(xb).argmax(dim=1)
            hit += (pred == yb).sum().item()
            total += yb.numel()
        return hit / max(total, 1)

    best = 0.0
    for ep in range(1, epochs + 1):
        probe.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = F.cross_entropy(probe(xb), yb)
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
    ap.add_argument("--max_train_groups", type=int, default=8000)
    ap.add_argument("--max_valid_groups", type=int, default=946)
    ap.add_argument("--feat_batch_size", type=int, default=16)

    ap.add_argument("--probe_hidden", type=int, default=0)
    ap.add_argument("--probe_batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-2)

    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_from_ckpt(args.ckpt, device=device)

    tok = AutoTokenizer.from_pretrained(cfg.get("backbone", "google/mt5-small"))
    max_doc_len = int(cfg.get("max_doc_len", 256))

    print("Extracting features (train)...")
    Xtr, ytr_lang, ytr_len = extract_features(
        model, tok, args.train_jsonl, device, args.max_train_groups, max_doc_len, args.feat_batch_size
    )
    print("Extracting features (valid)...")
    Xva, yva_lang, yva_len = extract_features(
        model, tok, args.valid_jsonl, device, args.max_valid_groups, max_doc_len, args.feat_batch_size
    )

    print(f"Feature shapes: Xtr={tuple(Xtr.shape)}, Xva={tuple(Xva.shape)}")

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
