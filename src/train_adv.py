from __future__ import annotations

import argparse
import json
import os
import time
import random
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import WikiLinguaGroupDataset, make_collate_fn
from model import LatentRendererModel


# -------------------------
# Basic losses
# -------------------------
def ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    b, l, v = logits.shape
    return F.cross_entropy(logits.view(b * l, v), labels.view(b * l), ignore_index=-100)


def info_nce_loss(zbar_en: torch.Tensor, zbar_zh: torch.Tensor, tau: float) -> torch.Tensor:
    sim = (zbar_en @ zbar_zh.t()) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_a = F.cross_entropy(sim, labels)
    loss_b = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_a + loss_b)


def variance_loss(z: torch.Tensor, target_std: float = 0.05, eps: float = 1e-4) -> torch.Tensor:
    # z: [B, D]
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(target_std - std))


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # x: [D, D]
    d = x.size(0)
    return x.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    # z: [B, D]
    z = z - z.mean(dim=0, keepdim=True)
    b = z.size(0)
    if b <= 1:
        return torch.tensor(0.0, device=z.device)
    cov = (z.t() @ z) / (b - 1)  # [D,D]
    return (off_diagonal(cov) ** 2).mean()


def varcov_regularizer(z: torch.Tensor, target_std: float) -> torch.Tensor:
    return variance_loss(z, target_std=target_std) + covariance_loss(z)


# -------------------------
# Gradient reversal for adversarial domain confusion
# -------------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


@torch.no_grad()
def eval_nll(model: LatentRendererModel, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    n = 0
    loss_en_sum = 0.0
    loss_zh_sum = 0.0
    for batch in dl:
        en_ids = batch["en_input_ids"].to(device)
        en_m = batch["en_attention_mask"].to(device)
        zh_ids = batch["zh_input_ids"].to(device)
        zh_m = batch["zh_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(en_ids, en_m, zh_ids, zh_m, labels)
        l_en = ce_loss_from_logits(out.logits_en, labels)
        l_zh = ce_loss_from_logits(out.logits_zh, labels)

        bs = en_ids.size(0)
        n += bs
        loss_en_sum += float(l_en.item()) * bs
        loss_zh_sum += float(l_zh.item()) * bs

    return {
        "nll_en": loss_en_sum / max(n, 1),
        "nll_zh": loss_zh_sum / max(n, 1),
        "nll": 0.5 * (loss_en_sum + loss_zh_sum) / max(n, 1),
    }


def save_checkpoint(path: str, model, lang_clf, optimizer, step: int, cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "lang_clf_state": lang_clf.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
    ap.add_argument("--run_dir", default="runs/adv0")

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

    # existing align
    ap.add_argument("--lambda_align", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.07)

    # NEW: language adversary + anti-collapse
    ap.add_argument("--lambda_langadv", type=float, default=0.5, help="0 disables language adversary")
    ap.add_argument("--lambda_varcov", type=float, default=1.0, help="0 disables var/cov regularizer")
    ap.add_argument("--var_target_std", type=float, default=0.05)

    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--max_train_examples", type=int, default=0)
    ap.add_argument("--max_valid_examples", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)

    cfg = vars(args)
    with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    tok = AutoTokenizer.from_pretrained(args.backbone)

    train_ds = WikiLinguaGroupDataset(args.train_jsonl, max_examples=(args.max_train_examples or None))
    valid_ds = WikiLinguaGroupDataset(args.valid_jsonl, max_examples=(args.max_valid_examples or None))
    collate = make_collate_fn(tok, max_doc_len=args.max_doc_len, max_sum_len=args.max_sum_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = LatentRendererModel(
        backbone_name=args.backbone,
        num_latents=args.num_latents,
        latent_dropout=args.latent_dropout,
        latent_noise_std=args.latent_noise_std,
    ).to(device)

    # language classifier on zbar (2-way: en vs zh)
    d_model = model.config.d_model
    lang_clf = nn.Linear(d_model, 2).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(lang_clf.parameters()), lr=args.lr)

    step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    log_path = os.path.join(args.run_dir, "logs.jsonl")
    ckpt_path = os.path.join(args.run_dir, "ckpt.pt")

    random.seed(42)
    torch.manual_seed(42)

    for epoch in range(args.epochs):
        model.train()
        lang_clf.train()

        for batch in train_dl:
            step += 1

            en_ids = batch["en_input_ids"].to(device)
            en_m = batch["en_attention_mask"].to(device)
            zh_ids = batch["zh_input_ids"].to(device)
            zh_m = batch["zh_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(en_ids, en_m, zh_ids, zh_m, labels)

            # render loss
            loss_en = ce_loss_from_logits(out.logits_en, labels)
            loss_zh = ce_loss_from_logits(out.logits_zh, labels)
            loss_render = 0.5 * (loss_en + loss_zh)

            # align loss (optional)
            loss_align = info_nce_loss(out.zbar_en, out.zbar_zh, tau=args.tau) if args.lambda_align > 0 else torch.tensor(0.0, device=device)

            # anti-collapse (var/cov) on both views
            if args.lambda_varcov > 0:
                loss_varcov = varcov_regularizer(out.zbar_en, args.var_target_std) + varcov_regularizer(out.zbar_zh, args.var_target_std)
            else:
                loss_varcov = torch.tensor(0.0, device=device)

            # language adversary (domain confusion)
            if args.lambda_langadv > 0:
                z_all = torch.cat([out.zbar_en, out.zbar_zh], dim=0)  # [2B, D]
                y_lang = torch.cat(
                    [
                        torch.zeros(out.zbar_en.size(0), dtype=torch.long, device=device),
                        torch.ones(out.zbar_zh.size(0), dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                # GRL with strength lambda_langadv
                logits_lang = lang_clf(grad_reverse(z_all, args.lambda_langadv))
                loss_lang = F.cross_entropy(logits_lang, y_lang)

                # monitoring accuracy
                with torch.no_grad():
                    acc_lang = (logits_lang.argmax(dim=1) == y_lang).float().mean().item()
            else:
                loss_lang = torch.tensor(0.0, device=device)
                acc_lang = 0.0

            loss = loss_render + args.lambda_align * loss_align + args.lambda_varcov * loss_varcov + loss_lang

            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(lang_clf.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 20 == 0:
                msg = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "loss_render": float(loss_render.item()),
                    "loss_align": float(loss_align.item()),
                    "loss_varcov": float(loss_varcov.item()),
                    "loss_lang": float(loss_lang.item()),
                    "lang_acc_batch": float(acc_lang),
                    "elapsed_sec": float(time.time() - t0),
                }
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(msg) + "\n")

            if step % args.eval_every == 0:
                diag = {"step": step, **eval_nll(model, valid_dl, device)}
                print("EVAL:", diag)
                with open(os.path.join(args.run_dir, "diag.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(diag) + "\n")
                save_checkpoint(ckpt_path, model, lang_clf, optimizer, step, cfg)

    save_checkpoint(ckpt_path, model, lang_clf, optimizer, step, cfg)
    print(f"Done. Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
