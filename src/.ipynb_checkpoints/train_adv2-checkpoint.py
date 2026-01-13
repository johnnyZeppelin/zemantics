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


def ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    b, l, v = logits.shape
    return F.cross_entropy(logits.view(b * l, v), labels.view(b * l), ignore_index=-100)


def info_nce_loss(zbar_en: torch.Tensor, zbar_zh: torch.Tensor, tau: float) -> torch.Tensor:
    sim = (zbar_en @ zbar_zh.t()) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


def variance_loss(z: torch.Tensor, target_std: float = 0.05, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(target_std - std))


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    d = x.size(0)
    return x.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    b = z.size(0)
    if b <= 1:
        return torch.tensor(0.0, device=z.device)
    cov = (z.t() @ z) / (b - 1)
    return (off_diagonal(cov) ** 2).mean()


def varcov_regularizer(z: torch.Tensor, target_std: float) -> torch.Tensor:
    return variance_loss(z, target_std=target_std) + covariance_loss(z)


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


def save_checkpoint(path: str, model, lang_clf, opt_model, opt_lang, step: int, cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "lang_clf_state": lang_clf.state_dict(),
            "opt_model_state": opt_model.state_dict(),
            "opt_lang_state": opt_lang.state_dict(),
            "config": cfg,
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
    ap.add_argument("--run_dir", default="runs/adv2_0")

    ap.add_argument("--backbone", default="google/mt5-small")
    ap.add_argument("--num_latents", type=int, default=16)
    ap.add_argument("--latent_dropout", type=float, default=0.1)
    ap.add_argument("--latent_noise_std", type=float, default=0.01)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--max_doc_len", type=int, default=256)
    ap.add_argument("--max_sum_len", type=int, default=64)

    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--max_train_examples", type=int, default=2000)
    ap.add_argument("--max_valid_examples", type=int, default=200)

    # losses
    ap.add_argument("--lambda_align", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=0.07)

    ap.add_argument("--lambda_langadv", type=float, default=1.0)
    ap.add_argument("--lang_steps", type=int, default=3, help="how many lang_clf updates per batch")

    ap.add_argument("--lambda_varcov", type=float, default=50.0)
    ap.add_argument("--var_target_std", type=float, default=0.05)

    # optim
    ap.add_argument("--lr_model", type=float, default=1e-4)
    ap.add_argument("--lr_lang", type=float, default=1e-2)

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

    d_model = model.config.d_model
    lang_clf = nn.Linear(d_model, 2).to(device)

    opt_model = torch.optim.AdamW(model.parameters(), lr=args.lr_model)
    opt_lang = torch.optim.AdamW(lang_clf.parameters(), lr=args.lr_lang)

    step = 0
    t0 = time.time()
    log_path = os.path.join(args.run_dir, "logs.jsonl")
    ckpt_path = os.path.join(args.run_dir, "ckpt.pt")

    random.seed(42)
    torch.manual_seed(42)

    opt_model.zero_grad(set_to_none=True)

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

            # ---------
            # 1) train lang_clf to be a strong adversary (detach z)
            # ---------
            with torch.no_grad():
                out0 = model(en_ids, en_m, zh_ids, zh_m, labels)
                z_all_det = torch.cat([out0.zbar_en, out0.zbar_zh], dim=0).detach()
                y_lang = torch.cat(
                    [
                        torch.zeros(out0.zbar_en.size(0), dtype=torch.long, device=device),
                        torch.ones(out0.zbar_zh.size(0), dtype=torch.long, device=device),
                    ],
                    dim=0,
                )

            for _ in range(args.lang_steps):
                opt_lang.zero_grad(set_to_none=True)
                logits_lang = lang_clf(z_all_det)
                loss_lang_clf = F.cross_entropy(logits_lang, y_lang)
                loss_lang_clf.backward()
                opt_lang.step()

            with torch.no_grad():
                acc_lang_clf = (logits_lang.argmax(dim=1) == y_lang).float().mean().item()

            # ---------
            # 2) update encoder/model with adversarial objective (freeze lang_clf)
            # ---------
            for p in lang_clf.parameters():
                p.requires_grad = False

            out = model(en_ids, en_m, zh_ids, zh_m, labels)

            loss_en = ce_loss_from_logits(out.logits_en, labels)
            loss_zh = ce_loss_from_logits(out.logits_zh, labels)
            loss_render = 0.5 * (loss_en + loss_zh)

            loss_align = info_nce_loss(out.zbar_en, out.zbar_zh, tau=args.tau) if args.lambda_align > 0 else torch.tensor(0.0, device=device)

            loss_varcov = torch.tensor(0.0, device=device)
            if args.lambda_varcov > 0:
                loss_varcov = varcov_regularizer(out.zbar_en, args.var_target_std) + varcov_regularizer(out.zbar_zh, args.var_target_std)

            z_all = torch.cat([out.zbar_en, out.zbar_zh], dim=0)
            logits_lang_adv = lang_clf(z_all)
            loss_lang_for_encoder = F.cross_entropy(logits_lang_adv, y_lang)

            # encoder wants to MAXIMIZE language loss, so subtract
            loss = (
                loss_render
                + args.lambda_align * loss_align
                + args.lambda_varcov * loss_varcov
                - args.lambda_langadv * loss_lang_for_encoder
            )

            (loss / args.grad_accum).backward()

            for p in lang_clf.parameters():
                p.requires_grad = True

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_model.step()
                opt_model.zero_grad(set_to_none=True)

            if step % 20 == 0:
                msg = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "loss_render": float(loss_render.item()),
                    "loss_align": float(loss_align.item()),
                    "loss_varcov": float(loss_varcov.item()),
                    "lang_loss_clf": float(loss_lang_clf.item()),
                    "lang_acc_clf_detached": float(acc_lang_clf),
                    "lang_loss_for_encoder": float(loss_lang_for_encoder.item()),
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
                save_checkpoint(ckpt_path, model, lang_clf, opt_model, opt_lang, step, cfg)

    save_checkpoint(ckpt_path, model, lang_clf, opt_model, opt_lang, step, cfg)
    print(f"Done. Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
