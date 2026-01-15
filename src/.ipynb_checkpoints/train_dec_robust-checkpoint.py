from __future__ import annotations

import argparse
import json
import os
import time
import random
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import WikiLinguaGroupDataset, make_collate_fn
from model import LatentRendererModel


def ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    b, l, v = logits.shape
    return F.cross_entropy(logits.reshape(b * l, v), labels.reshape(b * l), ignore_index=-100)


def set_requires_grad(m: nn.Module, flag: bool) -> None:
    for p in m.parameters():
        p.requires_grad_(flag)


def freeze_all(model: nn.Module) -> None:
    set_requires_grad(model, False)


def unfreeze_decoder_only(lrm: LatentRendererModel, train_lm_head: bool = True) -> None:
    """
    Freeze everything, then unfreeze decoder (and optionally lm_head).
    We keep encoder, bottleneck frozen to avoid changing z (and leakage).
    """
    freeze_all(lrm)

    # HF seq2seq model: encoder/decoder live inside lrm.model
    # For T5-like: lrm.model.get_decoder() exists
    dec = lrm.model.get_decoder()
    set_requires_grad(dec, True)

    # lm_head is used for vocab projection
    if train_lm_head and hasattr(lrm.model, "lm_head") and isinstance(lrm.model.lm_head, nn.Module):
        set_requires_grad(lrm.model.lm_head, True)

    # IMPORTANT: keep shared embeddings frozen by default (they affect encoder input too)
    if hasattr(lrm.model, "shared") and isinstance(lrm.model.shared, nn.Module):
        set_requires_grad(lrm.model.shared, False)


def maybe_untie_lm_head(lrm: LatentRendererModel) -> None:
    """
    Create a fresh lm_head (copy init weights) so we can train output projection
    without touching encoder-decoder shared embeddings.
    This is useful because T5 ties lm_head weights to shared embeddings.
    """
    if not hasattr(lrm.model, "lm_head"):
        return
    lm_head = lrm.model.lm_head
    if not isinstance(lm_head, nn.Linear):
        return

    d_model = lrm.config.d_model
    vocab = lrm.config.vocab_size

    new_head = nn.Linear(d_model, vocab, bias=False).to(next(lrm.parameters()).device)
    with torch.no_grad():
        # copy from existing lm_head (tied or not)
        new_head.weight.copy_(lm_head.weight.detach())

    lrm.model.lm_head = new_head


@torch.no_grad()
def encode_to_z_deterministic(
    lrm: LatentRendererModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Deterministic encoder+bottleneck forward (no dropout, no noise).
    Returns z: [B, K, D]
    """
    # Put encoder+bottleneck in eval to disable dropout
    enc = lrm.model.get_encoder()
    enc_was_training = enc.training
    bn_was_training = lrm.bottleneck.training

    enc.eval()
    lrm.bottleneck.eval()

    enc_out = enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    h = enc_out.last_hidden_state
    z = lrm.bottleneck(h, h_attn_mask=attention_mask)

    # restore states
    if enc_was_training:
        enc.train()
    if bn_was_training:
        lrm.bottleneck.train()

    return z


def noise_ramp(step: int, start_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if step <= start_step:
        return 0.0
    k = min(step - start_step, warmup_steps)
    return float(k) / float(warmup_steps)


@torch.no_grad()
def eval_nll(lrm: LatentRendererModel, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    lrm.eval()
    n = 0
    loss_en_sum = 0.0
    loss_zh_sum = 0.0

    for batch in dl:
        en_ids = batch["en_input_ids"].to(device)
        en_m = batch["en_attention_mask"].to(device)
        zh_ids = batch["zh_input_ids"].to(device)
        zh_m = batch["zh_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        z_en = encode_to_z_deterministic(lrm, en_ids, en_m)
        z_zh = encode_to_z_deterministic(lrm, zh_ids, zh_m)

        logits_en = lrm.decode_from_z(z_en, labels=labels)
        logits_zh = lrm.decode_from_z(z_zh, labels=labels)

        l_en = ce_loss_from_logits(logits_en, labels)
        l_zh = ce_loss_from_logits(logits_zh, labels)

        bs = en_ids.size(0)
        n += bs
        loss_en_sum += float(l_en.item()) * bs
        loss_zh_sum += float(l_zh.item()) * bs

    return {
        "nll_en": loss_en_sum / max(n, 1),
        "nll_zh": loss_zh_sum / max(n, 1),
        "nll": 0.5 * (loss_en_sum + loss_zh_sum) / max(n, 1),
    }


def save_checkpoint(
    path: str,
    lrm: LatentRendererModel,
    opt: torch.optim.Optimizer,
    step: int,
    cfg: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "step": step,
        "model_state": lrm.state_dict(),
        "opt_state": opt.state_dict(),
        "config": cfg,
    }
    torch.save(payload, path)


def try_load_checkpoint_model_only(
    resume_path: str,
    lrm: LatentRendererModel,
    device: torch.device,
) -> int:
    if not resume_path:
        return 0
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"--resume not found: {resume_path}")

    ckpt = torch.load(resume_path, map_location=device)
    if "model_state" in ckpt:
        lrm.load_state_dict(ckpt["model_state"], strict=True)
    else:
        # allow resuming from a pure model state dict
        lrm.load_state_dict(ckpt, strict=True)

    step = int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0
    print(f"[RESUME-MODEL-ONLY] loaded {resume_path}, step={step}")
    return step


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
    ap.add_argument("--run_dir", default="runs/dec_robust_smoke")
    ap.add_argument("--resume", default="", help="path to ckpt.pt (model_state supported)")

    ap.add_argument("--backbone", default="google/mt5-small")
    ap.add_argument("--num_latents", type=int, default=16)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--max_doc_len", type=int, default=256)
    ap.add_argument("--max_sum_len", type=int, default=64)

    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--max_train_examples", type=int, default=0, help="0 means all")
    ap.add_argument("--max_valid_examples", type=int, default=0, help="0 means all")

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    # Robustness knobs applied on Z before decoder
    ap.add_argument("--latent_dropout", type=float, default=0.1)
    ap.add_argument("--latent_noise_std", type=float, default=0.01)

    ap.add_argument("--noise_warmup_steps", type=int, default=500)
    ap.add_argument("--noise_warmup_start_step", type=int, default=-1, help="-1 means use resumed step as start")

    ap.add_argument("--untie_lm_head", action="store_true", help="train a fresh lm_head without touching shared embeddings")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.run_dir, exist_ok=True)
    cfg = vars(args)
    with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.backbone)
    train_ds = WikiLinguaGroupDataset(args.train_jsonl, max_examples=(None if args.max_train_examples <= 0 else args.max_train_examples))
    valid_ds = WikiLinguaGroupDataset(args.valid_jsonl, max_examples=(None if args.max_valid_examples <= 0 else args.max_valid_examples))
    collate = make_collate_fn(tok, max_doc_len=args.max_doc_len, max_sum_len=args.max_sum_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    lrm = LatentRendererModel(
        backbone_name=args.backbone,
        num_latents=args.num_latents,
        latent_dropout=0.0,     # we apply dropout ourselves on z
        latent_noise_std=0.0,   # we apply noise ourselves on z
    ).to(device)

    step = try_load_checkpoint_model_only(args.resume, lrm, device)

    if args.untie_lm_head:
        maybe_untie_lm_head(lrm)

    # decoder-only trainable params
    unfreeze_decoder_only(lrm, train_lm_head=True)

    trainable_params = [p for p in lrm.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters. Check decoder-only unfreeze logic.")

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    log_path = os.path.join(args.run_dir, "logs.jsonl")
    diag_path = os.path.join(args.run_dir, "diag.jsonl")
    ckpt_path = os.path.join(args.run_dir, "ckpt.pt")

    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    if args.noise_warmup_start_step < 0:
        noise_start = step
    else:
        noise_start = args.noise_warmup_start_step

    for epoch in range(args.epochs):
        lrm.train()  # decoder train mode

        for batch in train_dl:
            step += 1

            en_ids = batch["en_input_ids"].to(device)
            en_m = batch["en_attention_mask"].to(device)
            zh_ids = batch["zh_input_ids"].to(device)
            zh_m = batch["zh_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 1) Encode z deterministically under no_grad
            with torch.no_grad():
                z_en = encode_to_z_deterministic(lrm, en_ids, en_m)
                z_zh = encode_to_z_deterministic(lrm, zh_ids, zh_m)

            # 2) Apply robustness augmentation on z (dropout + noise), still no grad needed for z
            # dropout (force training=True to always apply)
            if args.latent_dropout > 0:
                z_en = F.dropout(z_en, p=args.latent_dropout, training=True)
                z_zh = F.dropout(z_zh, p=args.latent_dropout, training=True)

            # noise with warmup
            r = noise_ramp(step, noise_start, args.noise_warmup_steps)
            noise_std = args.latent_noise_std * r
            if noise_std > 0:
                z_en = z_en + torch.randn_like(z_en) * noise_std
                z_zh = z_zh + torch.randn_like(z_zh) * noise_std

            # 3) Decode and compute render loss
            logits_en = lrm.decode_from_z(z_en, labels=labels)
            logits_zh = lrm.decode_from_z(z_zh, labels=labels)

            loss_en = ce_loss_from_logits(logits_en, labels)
            loss_zh = ce_loss_from_logits(logits_zh, labels)
            loss = 0.5 * (loss_en + loss_zh)

            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % 20 == 0:
                msg = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "loss_en": float(loss_en.item()),
                    "loss_zh": float(loss_zh.item()),
                    "noise_std_eff": float(noise_std),
                    "elapsed_sec": float(time.time() - t0),
                }
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(msg) + "\n")

            if step % args.eval_every == 0:
                diag = {"step": step, **eval_nll(lrm, valid_dl, device)}
                print("EVAL:", diag)
                with open(diag_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(diag) + "\n")
                save_checkpoint(ckpt_path, lrm, opt, step, cfg)

    save_checkpoint(ckpt_path, lrm, opt, step, cfg)
    print(f"Done. Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
