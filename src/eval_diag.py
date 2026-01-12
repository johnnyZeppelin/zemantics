from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import WikiLinguaGroupDataset, make_collate_fn
from model import LatentRendererModel


def ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    b, l, v = logits.shape
    return F.cross_entropy(
        logits.view(b * l, v),
        labels.view(b * l),
        ignore_index=-100,
    )


@torch.no_grad()
def nll_from_z(model: LatentRendererModel, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute NLL given latent slots z and labels (teacher forcing).
    Returns scalar tensor.
    """
    logits = model.decode_from_z(z, labels=labels)
    return ce_loss_from_logits(logits, labels)


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[LatentRendererModel, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    backbone = cfg.get("backbone", "google/mt5-small")
    num_latents = int(cfg.get("num_latents", 16))
    latent_dropout = float(cfg.get("latent_dropout", 0.1))

    # Eval时关闭latent噪声，结果更稳定
    model = LatentRendererModel(
        backbone_name=backbone,
        num_latents=num_latents,
        latent_dropout=latent_dropout,
        latent_noise_std=0.0,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def make_derangement_perm(bsz: int, device: torch.device) -> torch.Tensor:
    """
    Simple derangement: shift by a random offset in [1, bsz-1].
    Works when bsz >= 2.
    """
    if bsz < 2:
        return torch.arange(bsz, device=device)
    offset = random.randint(1, bsz - 1)
    return (torch.arange(bsz, device=device) + offset) % bsz


@torch.no_grad()
def run_swap_and_ablation(
    model: LatentRendererModel,
    dl: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, float]:
    """
    Compute:
      - normal NLL for en_doc->en_sum and zh_doc->en_sum
      - swap delta NLL (batch derangement swap)
      - ablation delta NLL for z: zeros, mean, noise
    All metrics are averaged over examples.
    """
    total = 0

    sum_nll_en = 0.0
    sum_nll_zh = 0.0

    sum_swap_delta_en = 0.0
    sum_swap_delta_zh = 0.0

    # ablations: zeros, mean, noise
    sum_ablate_zero_delta_en = 0.0
    sum_ablate_mean_delta_en = 0.0
    sum_ablate_noise_delta_en = 0.0

    sum_ablate_zero_delta_zh = 0.0
    sum_ablate_mean_delta_zh = 0.0
    sum_ablate_noise_delta_zh = 0.0

    for bi, batch in enumerate(dl):
        if max_batches and bi >= max_batches:
            break

        en_ids = batch["en_input_ids"].to(device)
        en_m = batch["en_attention_mask"].to(device)
        zh_ids = batch["zh_input_ids"].to(device)
        zh_m = batch["zh_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        bsz = en_ids.size(0)
        total += bsz

        # Encode to latents (no noise)
        z_en = model.encode_to_z(en_ids, en_m, add_noise=False)
        z_zh = model.encode_to_z(zh_ids, zh_m, add_noise=False)

        # Normal NLL
        nll_en = nll_from_z(model, z_en, labels)
        nll_zh = nll_from_z(model, z_zh, labels)

        sum_nll_en += float(nll_en.item()) * bsz
        sum_nll_zh += float(nll_zh.item()) * bsz

        # Swap NLL (permute within batch)
        perm = make_derangement_perm(bsz, device=device)
        z_en_swap = z_en[perm]
        z_zh_swap = z_zh[perm]

        nll_en_swap = nll_from_z(model, z_en_swap, labels)
        nll_zh_swap = nll_from_z(model, z_zh_swap, labels)

        sum_swap_delta_en += float((nll_en_swap - nll_en).item()) * bsz
        sum_swap_delta_zh += float((nll_zh_swap - nll_zh).item()) * bsz

        # Ablation: zeros, mean, noise
        z_en_zero = torch.zeros_like(z_en)
        z_zh_zero = torch.zeros_like(z_zh)

        z_en_mean = z_en.mean(dim=0, keepdim=True).expand_as(z_en)
        z_zh_mean = z_zh.mean(dim=0, keepdim=True).expand_as(z_zh)

        z_en_noise = torch.randn_like(z_en)
        z_zh_noise = torch.randn_like(z_zh)

        nll_en_zero = nll_from_z(model, z_en_zero, labels)
        nll_en_mean = nll_from_z(model, z_en_mean, labels)
        nll_en_noise = nll_from_z(model, z_en_noise, labels)

        nll_zh_zero = nll_from_z(model, z_zh_zero, labels)
        nll_zh_mean = nll_from_z(model, z_zh_mean, labels)
        nll_zh_noise = nll_from_z(model, z_zh_noise, labels)

        sum_ablate_zero_delta_en += float((nll_en_zero - nll_en).item()) * bsz
        sum_ablate_mean_delta_en += float((nll_en_mean - nll_en).item()) * bsz
        sum_ablate_noise_delta_en += float((nll_en_noise - nll_en).item()) * bsz

        sum_ablate_zero_delta_zh += float((nll_zh_zero - nll_zh).item()) * bsz
        sum_ablate_mean_delta_zh += float((nll_zh_mean - nll_zh).item()) * bsz
        sum_ablate_noise_delta_zh += float((nll_zh_noise - nll_zh).item()) * bsz

    denom = max(total, 1)
    return {
        "nll_en": sum_nll_en / denom,
        "nll_zh": sum_nll_zh / denom,
        "swap_delta_en": sum_swap_delta_en / denom,
        "swap_delta_zh": sum_swap_delta_zh / denom,
        "ablate_zero_delta_en": sum_ablate_zero_delta_en / denom,
        "ablate_mean_delta_en": sum_ablate_mean_delta_en / denom,
        "ablate_noise_delta_en": sum_ablate_noise_delta_en / denom,
        "ablate_zero_delta_zh": sum_ablate_zero_delta_zh / denom,
        "ablate_mean_delta_zh": sum_ablate_mean_delta_zh / denom,
        "ablate_noise_delta_zh": sum_ablate_noise_delta_zh / denom,
    }


@torch.no_grad()
def compute_full_retrieval(
    model: LatentRendererModel,
    dl: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Full valid retrieval:
      compute zbar_en and zbar_zh for all examples, then NxN similarity.
    Metrics:
      - inv_top1_full
      - inv_top5_full
      - diag_sim_mean, offdiag_sim_mean
    """
    z_en_all: List[torch.Tensor] = []
    z_zh_all: List[torch.Tensor] = []

    for batch in dl:
        en_ids = batch["en_input_ids"].to(device)
        en_m = batch["en_attention_mask"].to(device)
        zh_ids = batch["zh_input_ids"].to(device)
        zh_m = batch["zh_attention_mask"].to(device)

        z_en = model.encode_to_z(en_ids, en_m, add_noise=False)
        z_zh = model.encode_to_z(zh_ids, zh_m, add_noise=False)

        zbar_en = model.pool_zbar(z_en).detach().cpu()
        zbar_zh = model.pool_zbar(z_zh).detach().cpu()

        z_en_all.append(zbar_en)
        z_zh_all.append(zbar_zh)

    zbar_en = torch.cat(z_en_all, dim=0)  # [N, D]
    zbar_zh = torch.cat(z_zh_all, dim=0)  # [N, D]
    n = zbar_en.size(0)

    sim = zbar_en @ zbar_zh.t()  # [N, N]

    # top-1, top-5 retrieval
    gt = torch.arange(n)

    top1 = sim.argmax(dim=1)
    inv_top1 = (top1 == gt).float().mean().item()

    k = min(5, n)
    topk = sim.topk(k=k, dim=1).indices  # [N, k]
    inv_top5 = (topk == gt.unsqueeze(1)).any(dim=1).float().mean().item()

    diag_mean = sim.diag().mean().item()
    offdiag_mean = (sim.sum() - sim.diag().sum()) / max(n * n - n, 1)
    offdiag_mean = float(offdiag_mean.item())

    return {
        "inv_top1_full": inv_top1,
        "inv_top5_full": inv_top5,
        "diag_sim_mean": diag_mean,
        "offdiag_sim_mean": offdiag_mean,
        "sim_margin": diag_mean - offdiag_mean,
        "n_valid": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="e.g. runs/exp0/ckpt.pt")
    ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_valid_examples", type=int, default=0, help="0 means full valid set")
    ap.add_argument("--max_batches", type=int, default=0, help="0 means use all batches for swap/ablation")
    ap.add_argument("--out", default="", help="Output json path. Default: <run_dir>/diag_full.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_from_ckpt(args.ckpt, device=device)

    backbone = cfg.get("backbone", "google/mt5-small")
    tok = AutoTokenizer.from_pretrained(backbone)

    valid_ds = WikiLinguaGroupDataset(args.valid_jsonl, max_examples=(args.max_valid_examples or None))
    collate = make_collate_fn(tok, max_doc_len=int(cfg.get("max_doc_len", 256)), max_sum_len=int(cfg.get("max_sum_len", 64)))
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # swap + ablation
    random.seed(42)
    torch.manual_seed(42)

    diag1 = run_swap_and_ablation(model, valid_dl, device=device, max_batches=args.max_batches)

    # full retrieval
    diag2 = compute_full_retrieval(model, valid_dl, device=device)

    diag = {"ckpt": args.ckpt, **diag1, **diag2}

    run_dir = os.path.dirname(args.ckpt)
    out_path = args.out or os.path.join(run_dir, "diag_full.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print("=== DIAG RESULTS ===")
    print(json.dumps(diag, indent=2))
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
