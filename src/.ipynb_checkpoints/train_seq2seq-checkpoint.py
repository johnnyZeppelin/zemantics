from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from dataset import WikiLinguaGroupDataset, make_collate_fn


@torch.no_grad()
def eval_nll_seq2seq(model, dl: DataLoader, device: torch.device) -> Dict[str, float]:
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

        out_en = model(input_ids=en_ids, attention_mask=en_m, labels=labels, use_cache=False, return_dict=True)
        out_zh = model(input_ids=zh_ids, attention_mask=zh_m, labels=labels, use_cache=False, return_dict=True)

        # HF loss already ignores -100 labels
        l_en = out_en.loss
        l_zh = out_zh.loss
        l = 0.5 * (l_en + l_zh)

        bs = en_ids.size(0)
        n += bs
        loss_sum += float(l.item()) * bs
        loss_en_sum += float(l_en.item()) * bs
        loss_zh_sum += float(l_zh.item()) * bs

    return {
        "nll": loss_sum / max(n, 1),
        "nll_en": loss_en_sum / max(n, 1),
        "nll_zh": loss_zh_sum / max(n, 1),
    }


def save_checkpoint(path: str, model, optimizer, step: int, cfg: Dict[str, Any]) -> None:
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
    ap.add_argument("--run_dir", default="runs/seq2seq0")

    ap.add_argument("--backbone", default="google/mt5-small")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--max_doc_len", type=int, default=256)
    ap.add_argument("--max_sum_len", type=int, default=64)

    ap.add_argument("--eval_every", type=int, default=200)
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

    model = AutoModelForSeq2SeqLM.from_pretrained(args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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

            out_en = model(input_ids=en_ids, attention_mask=en_m, labels=labels, use_cache=False, return_dict=True)
            out_zh = model(input_ids=zh_ids, attention_mask=zh_m, labels=labels, use_cache=False, return_dict=True)

            loss_en = out_en.loss
            loss_zh = out_zh.loss
            loss = 0.5 * (loss_en + loss_zh)

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
                    "loss_en": float(loss_en.item()),
                    "loss_zh": float(loss_zh.item()),
                    "elapsed_sec": elapsed,
                }
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(msg) + "\n")

            if step % args.eval_every == 0:
                nll = eval_nll_seq2seq(model, valid_dl, device)
                diag = {"step": step, **nll}
                print("EVAL:", diag)
                with open(os.path.join(args.run_dir, "diag.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(diag) + "\n")

                save_checkpoint(ckpt_path, model, optimizer, step, cfg)

    save_checkpoint(ckpt_path, model, optimizer, step, cfg)
    print(f"Done. Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
