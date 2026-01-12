from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Any, Tuple, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from dataset import WikiLinguaGroupDataset, make_collate_fn


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    backbone = cfg.get("backbone", "google/mt5-small")

    model = AutoModelForSeq2SeqLM.from_pretrained(backbone)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def mean_pool_hidden(h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """
    h: [B, T, D]
    attn_mask: [B, T] (1 keep, 0 pad)
    return zbar: [B, D] L2-normalized
    """
    m = attn_mask.unsqueeze(-1).float()  # [B,T,1]
    denom = m.sum(dim=1).clamp(min=1.0)  # [B,1]
    pooled = (h * m).sum(dim=1) / denom  # [B,D]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled


def make_derangement_perm(bsz: int, device: torch.device) -> torch.Tensor:
    if bsz < 2:
        return torch.arange(bsz, device=device)
    offset = random.randint(1, bsz - 1)
    return (torch.arange(bsz, device=device) + offset) % bsz


@torch.no_grad()
def nll_from_encoder_outputs(model, h: torch.Tensor, attn_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    enc_out = BaseModelOutput(last_hidden_state=h)
    out = model(
        encoder_outputs=enc_out,
        encoder_attention_mask=attn_mask,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    return out.loss


@torch.no_grad()
def run_swap_and_ablation(model, dl: DataLoader, device: torch.device, max_batches: int = 0) -> Dict[str, float]:
    total = 0

    sum_nll_en = 0.0
    sum_nll_zh = 0.0

    sum_swap_delta_en = 0.0
    sum_swap_delta_zh = 0.0

    sum_ablate_zero_delta_en = 0.0
    sum_ablate_mean_delta_en = 0.0
    sum_ablate_noise_delta_en = 0.0

    sum_ablate_zero_delta_zh = 0.0
    sum_ablate_mean_delta_zh = 0.0
    sum_ablate_noise_delta_zh = 0.0

    encoder = model.get_encoder()

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

        h_en = encoder(input_ids=en_ids, attention_mask=en_m, return_dict=True).last_hidden_state
        h_zh = encoder(input_ids=zh_ids, attention_mask=zh_m, return_dict=True).last_hidden_state

        nll_en = nll_from_encoder_outputs(model, h_en, en_m, labels)
        nll_zh = nll_from_encoder_outputs(model, h_zh, zh_m, labels)

        sum_nll_en += float(nll_en.item()) * bsz
        sum_nll_zh += float(nll_zh.item()) * bsz

        perm = make_derangement_perm(bsz, device=device)

        h_en_swap = h_en[perm]
        en_m_swap = en_m[perm]
        h_zh_swap = h_zh[perm]
        zh_m_swap = zh_m[perm]

        nll_en_swap = nll_from_encoder_outputs(model, h_en_swap, en_m_swap, labels)
        nll_zh_swap = nll_from_encoder_outputs(model, h_zh_swap, zh_m_swap, labels)

        sum_swap_delta_en += float((nll_en_swap - nll_en).item()) * bsz
        sum_swap_delta_zh += float((nll_zh_swap - nll_zh).item()) * bsz

        # ablations
        h_en_zero = torch.zeros_like(h_en)
        h_zh_zero = torch.zeros_like(h_zh)

        h_en_mean = h_en.mean(dim=0, keepdim=True).expand_as(h_en)
        h_zh_mean = h_zh.mean(dim=0, keepdim=True).expand_as(h_zh)

        h_en_noise = torch.randn_like(h_en)
        h_zh_noise = torch.randn_like(h_zh)

        nll_en_zero = nll_from_encoder_outputs(model, h_en_zero, en_m, labels)
        nll_en_mean = nll_from_encoder_outputs(model, h_en_mean, en_m, labels)
        nll_en_noise = nll_from_encoder_outputs(model, h_en_noise, en_m, labels)

        nll_zh_zero = nll_from_encoder_outputs(model, h_zh_zero, zh_m, labels)
        nll_zh_mean = nll_from_encoder_outputs(model, h_zh_mean, zh_m, labels)
        nll_zh_noise = nll_from_encoder_outputs(model, h_zh_noise, zh_m, labels)

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
def compute_full_retrieval(model, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    encoder = model.get_encoder()

    z_en_all: List[torch.Tensor] = []
    z_zh_all: List[torch.Tensor] = []

    for batch in dl:
        en_ids = batch["en_input_ids"].to(device)
        en_m = batch["en_attention_mask"].to(device)
        zh_ids = batch["zh_input_ids"].to(device)
        zh_m = batch["zh_attention_mask"].to(device)

        h_en = encoder(input_ids=en_ids, attention_mask=en_m, return_dict=True).last_hidden_state
        h_zh = encoder(input_ids=zh_ids, attention_mask=zh_m, return_dict=True).last_hidden_state

        zbar_en = mean_pool_hidden(h_en, en_m).detach().cpu()
        zbar_zh = mean_pool_hidden(h_zh, zh_m).detach().cpu()

        z_en_all.append(zbar_en)
        z_zh_all.append(zbar_zh)

    zbar_en = torch.cat(z_en_all, dim=0)  # [N,D]
    zbar_zh = torch.cat(z_zh_all, dim=0)
    n = zbar_en.size(0)

    sim = zbar_en @ zbar_zh.t()

    gt = torch.arange(n)
    top1 = sim.argmax(dim=1)
    inv_top1 = (top1 == gt).float().mean().item()

    k = min(5, n)
    topk = sim.topk(k=k, dim=1).indices
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_valid_examples", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_from_ckpt(args.ckpt, device)

    tok = AutoTokenizer.from_pretrained(cfg.get("backbone", "google/mt5-small"))
    valid_ds = WikiLinguaGroupDataset(args.valid_jsonl, max_examples=(args.max_valid_examples or None))
    collate = make_collate_fn(
        tok,
        max_doc_len=int(cfg.get("max_doc_len", 256)),
        max_sum_len=int(cfg.get("max_sum_len", 64)),
    )
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    random.seed(42)
    torch.manual_seed(42)

    diag1 = run_swap_and_ablation(model, valid_dl, device=device, max_batches=args.max_batches)
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
