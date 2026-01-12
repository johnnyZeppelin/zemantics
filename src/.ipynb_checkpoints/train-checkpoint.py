from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import WikiLinguaGroupDataset, make_collate_fn
from model import LatentRendererModel


def ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, L, V]
    labels: [B, L] with -100 masked pads
    """
    b, l, v = logits.shape
    return F.cross_entropy(
        logits.view(b * l, v),
        labels.view(b * l),
        ignore_index=-100,
    )


def info_nce_loss(zbar_en: torch.Tensor, zbar_zh: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    zbar_*: [B, D], assumed L2-normalized.
    InfoNCE with positives on diagonal.
    """
    # Similarity matrix [B, B]
    sim = (zbar_en @ zbar_zh.t()) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_a = F.cross_entropy(sim, labels)
    loss_b = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_a + loss_b)


@torch.no_grad()
def eval_invariance_top1(model: LatentRendererModel, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Very basic invariance diagnostic:
    For each batch, compute zbar_en and zbar_zh, then retrieval top-1 accuracy:
      for each i, argmax_j zbar_en[i] dot zbar_zh[j] should be i.
    """
    model.eval()
    total = 0
    hit1 = 0

    for batch in dl:
        en_ids = batch["en_input_ids"].to(device)
        en_m = batch["en_attention_mask"].to(device)
        zh_ids = batch["zh_input_ids"].to(device)
        zh_m = batch["zh_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(
            en_input_ids=en_ids,
            en_attention_mask=en_m,
            zh_input_ids=zh_ids,
            zh_attention_mask=zh_m,
            labels=labels,
        )
        # retrieval
        sim = out.zbar_en @ out.zbar_zh.t()  # [B,B]
        pred = sim.argmax(dim=1)  # [B]
        gt = torch.arange(sim.size(0), device=device)
        hit1 += (pred == gt).sum().item()
        total += sim.size(0)

    return {"inv_top1": hit1 / max(total, 1)}


@torch.no_grad()
def eval_nll(model: LatentRendererModel, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    n = 0
    loss_sum = 0.0
    loss_en_sum = 0.0
    loss_zh_sum = 0.0

    for batch in dl:
        en_ids = batch["en_input_ids"].to(device)
        en_m = batch["en_attention_mask"].to(device)
        zh_ids = batch["zh_input_ids"].to(device)
        zh_m = batch["zh_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(
            en_input_ids=en_ids,
            en_attention_mask=en_m,
            zh_input_ids=zh_ids,
            zh_attention_mask=zh_m,
            labels=labels,
        )
        l_en = ce_loss_from_logits(out.logits_en, labels)
        l_zh = ce_loss_from_logits(out.logits_zh, labels)
        l = 0.5 * (l_en + l_zh)

        bs = en_ids.size(0)
        n += bs
        loss_sum += l.item() * bs
        loss_en_sum += l_en.item() * bs
        loss_zh_sum += l_zh.item() * bs

    return {
        "nll": loss_sum / max(n, 1),
        "nll_en": loss_en_sum / max(n, 1),
        "nll_zh": loss_zh_sum / max(n, 1),
    }


def save_checkpoint(path: str, model: LatentRendererModel, optimizer: torch.optim.Optimizer, step: int, cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
    ap.add_argument("--run_dir", default="runs/exp0")

    ap.add_argument("--backbone", default="google/mt5-small")
    ap.add_argument("--num_latents", type=int, default=16)
    ap.add_argument("--latent_dropout", type=float, default=0.1)
    ap.add_argument("--latent_noise_std", type=float, default=0.01)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--max_doc_len", type=int, default=256)
    ap.add_argument("--max_sum_len", type=int, default=64)

    ap.add_argument("--lambda_align", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.07)

    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--max_train_examples", type=int, default=0, help="0 means full dataset")
    ap.add_argument("--max_valid_examples", type=int, default=0, help="0 means full dataset")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)

    cfg = vars(args)
    with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)

    # Datasets
    train_ds = WikiLinguaGroupDataset(args.train_jsonl, max_examples=(args.max_train_examples or None))
    valid_ds = WikiLinguaGroupDataset(args.valid_jsonl, max_examples=(args.max_valid_examples or None))

    collate = make_collate_fn(tok, max_doc_len=args.max_doc_len, max_sum_len=args.max_sum_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Model
    model = LatentRendererModel(
        backbone_name=args.backbone,
        num_latents=args.num_latents,
        latent_dropout=args.latent_dropout,
        latent_noise_std=args.latent_noise_std,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training
    step = 0
    t0 = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    log_path = os.path.join(args.run_dir, "logs.jsonl")
    ckpt_path = os.path.join(args.run_dir, "ckpt.pt")

    for epoch in range(args.epochs):
        for batch in train_dl:
            step += 1
            en_ids = batch["en_input_ids"].to(device)
            en_m = batch["en_attention_mask"].to(device)
            zh_ids = batch["zh_input_ids"].to(device)
            zh_m = batch["zh_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(
                en_input_ids=en_ids,
                en_attention_mask=en_m,
                zh_input_ids=zh_ids,
                zh_attention_mask=zh_m,
                labels=labels,
            )

            loss_en = ce_loss_from_logits(out.logits_en, labels)
            loss_zh = ce_loss_from_logits(out.logits_zh, labels)
            loss_render = 0.5 * (loss_en + loss_zh)

            loss_align = info_nce_loss(out.zbar_en, out.zbar_zh, tau=args.tau)
            loss = loss_render + args.lambda_align * loss_align

            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 20 == 0:
                elapsed = time.time() - t0
                msg = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "loss_render": float(loss_render.item()),
                    "loss_align": float(loss_align.item()),
                    "elapsed_sec": elapsed,
                }
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(msg) + "\n")

            if step % args.eval_every == 0:
                # quick eval
                nll = eval_nll(model, valid_dl, device)
                inv = eval_invariance_top1(model, valid_dl, device)
                diag = {"step": step, **nll, **inv}
                print("EVAL:", diag)
                with open(os.path.join(args.run_dir, "diag.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(diag) + "\n")

                save_checkpoint(ckpt_path, model, optimizer, step, cfg)

    save_checkpoint(ckpt_path, model, optimizer, step, cfg)
    print(f"Done. Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
