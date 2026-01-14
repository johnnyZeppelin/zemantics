# # from __future__ import annotations

# # import argparse
# # import json
# # import os
# # import time
# # import random
# # from typing import Dict, Any

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader
# # from transformers import AutoTokenizer

# # from dataset import WikiLinguaGroupDataset, make_collate_fn
# # from model import LatentRendererModel


# # def ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
# #     b, l, v = logits.shape
# #     return F.cross_entropy(logits.view(b * l, v), labels.view(b * l), ignore_index=-100)


# # def info_nce_loss(zbar_en: torch.Tensor, zbar_zh: torch.Tensor, tau: float) -> torch.Tensor:
# #     sim = (zbar_en @ zbar_zh.t()) / tau
# #     labels = torch.arange(sim.size(0), device=sim.device)
# #     return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


# # def variance_loss(z: torch.Tensor, target_std: float = 0.05, eps: float = 1e-4) -> torch.Tensor:
# #     std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
# #     return torch.mean(F.relu(target_std - std))


# # def off_diagonal(x: torch.Tensor) -> torch.Tensor:
# #     d = x.size(0)
# #     return x.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()


# # def covariance_loss(z: torch.Tensor) -> torch.Tensor:
# #     z = z - z.mean(dim=0, keepdim=True)
# #     b = z.size(0)
# #     if b <= 1:
# #         return torch.tensor(0.0, device=z.device)
# #     cov = (z.t() @ z) / (b - 1)
# #     return (off_diagonal(cov) ** 2).mean()


# # def varcov_regularizer(z: torch.Tensor, target_std: float) -> torch.Tensor:
# #     return variance_loss(z, target_std=target_std) + covariance_loss(z)


# # class GradReverseFn(torch.autograd.Function):
# #     @staticmethod
# #     def forward(ctx, x: torch.Tensor, alpha: float):
# #         ctx.alpha = alpha
# #         return x.view_as(x)

# #     @staticmethod
# #     def backward(ctx, grad_output):
# #         return -ctx.alpha * grad_output, None


# # def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
# #     return GradReverseFn.apply(x, alpha)


# # @torch.no_grad()
# # def eval_nll(model: LatentRendererModel, dl: DataLoader, device: torch.device) -> Dict[str, float]:
# #     model.eval()
# #     n = 0
# #     loss_en_sum = 0.0
# #     loss_zh_sum = 0.0
# #     for batch in dl:
# #         en_ids = batch["en_input_ids"].to(device)
# #         en_m = batch["en_attention_mask"].to(device)
# #         zh_ids = batch["zh_input_ids"].to(device)
# #         zh_m = batch["zh_attention_mask"].to(device)
# #         labels = batch["labels"].to(device)

# #         out = model(en_ids, en_m, zh_ids, zh_m, labels)
# #         l_en = ce_loss_from_logits(out.logits_en, labels)
# #         l_zh = ce_loss_from_logits(out.logits_zh, labels)

# #         bs = en_ids.size(0)
# #         n += bs
# #         loss_en_sum += float(l_en.item()) * bs
# #         loss_zh_sum += float(l_zh.item()) * bs

# #     return {
# #         "nll_en": loss_en_sum / max(n, 1),
# #         "nll_zh": loss_zh_sum / max(n, 1),
# #         "nll": 0.5 * (loss_en_sum + loss_zh_sum) / max(n, 1),
# #     }


# # def save_checkpoint(path: str, model, lang_clf, opt_model, opt_lang, step: int, cfg: Dict[str, Any]) -> None:
# #     os.makedirs(os.path.dirname(path), exist_ok=True)
# #     torch.save(
# #         {
# #             "step": step,
# #             "model_state": model.state_dict(),
# #             "lang_clf_state": lang_clf.state_dict(),
# #             "opt_model_state": opt_model.state_dict(),
# #             "opt_lang_state": opt_lang.state_dict(),
# #             "config": cfg,
# #         },
# #         path,
# #     )


# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--train_jsonl", default="data/groups_train.jsonl")
# #     ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
# #     ap.add_argument("--run_dir", default="runs/adv4_smoke")

# #     ap.add_argument("--backbone", default="google/mt5-small")
# #     ap.add_argument("--num_latents", type=int, default=16)
# #     ap.add_argument("--latent_dropout", type=float, default=0.1)
# #     ap.add_argument("--latent_noise_std", type=float, default=0.01)

# #     ap.add_argument("--batch_size", type=int, default=4)
# #     ap.add_argument("--grad_accum", type=int, default=8)
# #     ap.add_argument("--epochs", type=int, default=1)

# #     ap.add_argument("--max_doc_len", type=int, default=256)
# #     ap.add_argument("--max_sum_len", type=int, default=64)

# #     ap.add_argument("--eval_every", type=int, default=400)
# #     ap.add_argument("--max_train_examples", type=int, default=2000)
# #     ap.add_argument("--max_valid_examples", type=int, default=200)

# #     # core losses
# #     ap.add_argument("--lambda_align", type=float, default=0.5)
# #     ap.add_argument("--tau", type=float, default=0.07)

# #     ap.add_argument("--lambda_varcov", type=float, default=10.0)
# #     ap.add_argument("--var_target_std", type=float, default=0.05)

# #     # mean controls (NOW use SUM, so keep lambdas small)
# #     ap.add_argument("--lambda_mean", type=float, default=0.1)
# #     ap.add_argument("--lambda_mean_diff", type=float, default=0.1)

# #     # NEW: paired closeness
# #     ap.add_argument("--lambda_pair", type=float, default=0.2, help="MSE(z_en, z_zh)")

# #     # GRL language adversary
# #     ap.add_argument("--lambda_lang", type=float, default=1.0, help="weight of language CE with GRL")
# #     ap.add_argument("--adv_start_step", type=int, default=300)
# #     ap.add_argument("--grl_alpha", type=float, default=1.0, help="GRL strength (gradient multiplier)")
# #     ap.add_argument("--grl_warmup", type=int, default=200, help="linear warmup steps after adv_start_step")

# #     # optim
# #     ap.add_argument("--lr_model", type=float, default=1e-4)
# #     ap.add_argument("--lr_lang", type=float, default=1e-3)

# #     args = ap.parse_args()
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #     os.makedirs(args.run_dir, exist_ok=True)
# #     cfg = vars(args)
# #     with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
# #         json.dump(cfg, f, indent=2)

# #     tok = AutoTokenizer.from_pretrained(args.backbone)
# #     train_ds = WikiLinguaGroupDataset(args.train_jsonl, max_examples=(args.max_train_examples or None))
# #     valid_ds = WikiLinguaGroupDataset(args.valid_jsonl, max_examples=(args.max_valid_examples or None))
# #     collate = make_collate_fn(tok, max_doc_len=args.max_doc_len, max_sum_len=args.max_sum_len)

# #     train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
# #     valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

# #     model = LatentRendererModel(
# #         backbone_name=args.backbone,
# #         num_latents=args.num_latents,
# #         latent_dropout=args.latent_dropout,
# #         latent_noise_std=args.latent_noise_std,
# #     ).to(device)

# #     d_model = model.config.d_model
# #     lang_clf = nn.Sequential(
# #         nn.Linear(d_model, d_model),
# #         nn.ReLU(),
# #         nn.Linear(d_model, 2),
# #     ).to(device)

# #     opt_model = torch.optim.AdamW(model.parameters(), lr=args.lr_model)
# #     opt_lang = torch.optim.AdamW(lang_clf.parameters(), lr=args.lr_lang)

# #     step = 0
# #     t0 = time.time()
# #     log_path = os.path.join(args.run_dir, "logs.jsonl")
# #     ckpt_path = os.path.join(args.run_dir, "ckpt.pt")

# #     random.seed(42)
# #     torch.manual_seed(42)

# #     opt_model.zero_grad(set_to_none=True)
# #     opt_lang.zero_grad(set_to_none=True)

# #     for epoch in range(args.epochs):
# #         model.train()
# #         lang_clf.train()

# #         for batch in train_dl:
# #             step += 1
# #             en_ids = batch["en_input_ids"].to(device)
# #             en_m = batch["en_attention_mask"].to(device)
# #             zh_ids = batch["zh_input_ids"].to(device)
# #             zh_m = batch["zh_attention_mask"].to(device)
# #             labels = batch["labels"].to(device)

# #             out = model(en_ids, en_m, zh_ids, zh_m, labels)
# #             z_en = out.zbar_en
# #             z_zh = out.zbar_zh
# #             z_all = torch.cat([z_en, z_zh], dim=0)

# #             # language labels
# #             y_lang = torch.cat(
# #                 [
# #                     torch.zeros(z_en.size(0), dtype=torch.long, device=device),
# #                     torch.ones(z_zh.size(0), dtype=torch.long, device=device),
# #                 ],
# #                 dim=0,
# #             )

# #             # render loss
# #             loss_en = ce_loss_from_logits(out.logits_en, labels)
# #             loss_zh = ce_loss_from_logits(out.logits_zh, labels)
# #             loss_render = 0.5 * (loss_en + loss_zh)

# #             # align loss
# #             loss_align = info_nce_loss(z_en, z_zh, tau=args.tau) if args.lambda_align > 0 else torch.tensor(0.0, device=device)

# #             # var/cov regularizer
# #             loss_varcov = torch.tensor(0.0, device=device)
# #             if args.lambda_varcov > 0:
# #                 loss_varcov = varcov_regularizer(z_en, args.var_target_std) + varcov_regularizer(z_zh, args.var_target_std)

# #             # mean penalties (use SUM so it matters)
# #             mu_all = z_all.mean(dim=0)
# #             mu_en = z_en.mean(dim=0)
# #             mu_zh = z_zh.mean(dim=0)
# #             loss_mean = (mu_all ** 2).sum()
# #             loss_mean_diff = ((mu_en - mu_zh) ** 2).sum()

# #             # NEW: paired closeness, directly kills language-private component
# #             loss_pair = ((z_en - z_zh) ** 2).mean()

# #             # GRL adversary
# #             loss_lang = torch.tensor(0.0, device=device)
# #             acc_lang = 0.0
# #             if step >= args.adv_start_step and args.lambda_lang > 0:
# #                 # warmup alpha
# #                 k = min(max(step - args.adv_start_step, 0), args.grl_warmup)
# #                 alpha = args.grl_alpha * (k / max(args.grl_warmup, 1))
# #                 z_rev = grad_reverse(z_all, alpha)
# #                 logits_lang = lang_clf(z_rev)
# #                 loss_lang = F.cross_entropy(logits_lang, y_lang)
# #                 with torch.no_grad():
# #                     acc_lang = (logits_lang.argmax(dim=1) == y_lang).float().mean().item()

# #             loss = (
# #                 loss_render
# #                 + args.lambda_align * loss_align
# #                 + args.lambda_varcov * loss_varcov
# #                 + args.lambda_mean * loss_mean
# #                 + args.lambda_mean_diff * loss_mean_diff
# #                 + args.lambda_pair * loss_pair
# #                 + args.lambda_lang * loss_lang
# #             )

# #             (loss / args.grad_accum).backward()

# #             if step % args.grad_accum == 0:
# #                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# #                 opt_model.step()
# #                 opt_lang.step()
# #                 opt_model.zero_grad(set_to_none=True)
# #                 opt_lang.zero_grad(set_to_none=True)

# #             if step % 20 == 0:
# #                 msg = {
# #                     "step": step,
# #                     "epoch": epoch,
# #                     "loss": float(loss.item()),
# #                     "loss_render": float(loss_render.item()),
# #                     "loss_align": float(loss_align.item()),
# #                     "loss_varcov": float(loss_varcov.item()),
# #                     "loss_mean": float(loss_mean.item()),
# #                     "loss_mean_diff": float(loss_mean_diff.item()),
# #                     "loss_pair": float(loss_pair.item()),
# #                     "loss_lang": float(loss_lang.item()),
# #                     "lang_acc_batch": float(acc_lang),
# #                     "elapsed_sec": float(time.time() - t0),
# #                 }
# #                 print(msg)
# #                 with open(log_path, "a", encoding="utf-8") as f:
# #                     f.write(json.dumps(msg) + "\n")

# #             if step % args.eval_every == 0:
# #                 diag = {"step": step, **eval_nll(model, valid_dl, device)}
# #                 print("EVAL:", diag)
# #                 with open(os.path.join(args.run_dir, "diag.jsonl"), "a", encoding="utf-8") as f:
# #                     f.write(json.dumps(diag) + "\n")
# #                 save_checkpoint(ckpt_path, model, lang_clf, opt_model, opt_lang, step, cfg)

# #     save_checkpoint(ckpt_path, model, lang_clf, opt_model, opt_lang, step, cfg)
# #     print(f"Done. Saved checkpoint to {ckpt_path}")


# # if __name__ == "__main__":
# #     main()


# from __future__ import annotations

# import argparse
# import json
# import os
# import time
# import random
# from typing import Dict, Any, Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer

# from dataset import WikiLinguaGroupDataset, make_collate_fn
# from model import LatentRendererModel


# def ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#     b, l, v = logits.shape
#     return F.cross_entropy(logits.view(b * l, v), labels.view(b * l), ignore_index=-100)


# def info_nce_loss(zbar_en: torch.Tensor, zbar_zh: torch.Tensor, tau: float) -> torch.Tensor:
#     sim = (zbar_en @ zbar_zh.t()) / tau
#     labels = torch.arange(sim.size(0), device=sim.device)
#     return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


# def variance_loss(z: torch.Tensor, target_std: float = 0.05, eps: float = 1e-4) -> torch.Tensor:
#     std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
#     return torch.mean(F.relu(target_std - std))


# def off_diagonal(x: torch.Tensor) -> torch.Tensor:
#     d = x.size(0)
#     return x.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()


# def covariance_loss(z: torch.Tensor) -> torch.Tensor:
#     z = z - z.mean(dim=0, keepdim=True)
#     b = z.size(0)
#     if b <= 1:
#         return torch.tensor(0.0, device=z.device)
#     cov = (z.t() @ z) / (b - 1)
#     return (off_diagonal(cov) ** 2).mean()


# def varcov_regularizer(z: torch.Tensor, target_std: float) -> torch.Tensor:
#     return variance_loss(z, target_std=target_std) + covariance_loss(z)


# class GradReverseFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x: torch.Tensor, alpha: float):
#         ctx.alpha = alpha
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return -ctx.alpha * grad_output, None


# def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
#     return GradReverseFn.apply(x, alpha)


# @torch.no_grad()
# def eval_nll(model: LatentRendererModel, dl: DataLoader, device: torch.device) -> Dict[str, float]:
#     model.eval()
#     n = 0
#     loss_en_sum = 0.0
#     loss_zh_sum = 0.0
#     for batch in dl:
#         en_ids = batch["en_input_ids"].to(device)
#         en_m = batch["en_attention_mask"].to(device)
#         zh_ids = batch["zh_input_ids"].to(device)
#         zh_m = batch["zh_attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         out = model(en_ids, en_m, zh_ids, zh_m, labels)
#         l_en = ce_loss_from_logits(out.logits_en, labels)
#         l_zh = ce_loss_from_logits(out.logits_zh, labels)

#         bs = en_ids.size(0)
#         n += bs
#         loss_en_sum += float(l_en.item()) * bs
#         loss_zh_sum += float(l_zh.item()) * bs

#     return {
#         "nll_en": loss_en_sum / max(n, 1),
#         "nll_zh": loss_zh_sum / max(n, 1),
#         "nll": 0.5 * (loss_en_sum + loss_zh_sum) / max(n, 1),
#     }


# def save_checkpoint(path: str, model, lang_clf, opt_model, opt_lang, step: int, cfg: Dict[str, Any]) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     torch.save(
#         {
#             "step": step,
#             "model_state": model.state_dict(),
#             "lang_clf_state": lang_clf.state_dict(),
#             "opt_model_state": opt_model.state_dict(),
#             "opt_lang_state": opt_lang.state_dict(),
#             "config": cfg,
#         },
#         path,
#     )


# def try_load_checkpoint(
#     resume_path: str,
#     model: nn.Module,
#     lang_clf: nn.Module,
#     opt_model: torch.optim.Optimizer,
#     opt_lang: torch.optim.Optimizer,
#     device: torch.device,
# ) -> int:
#     if not resume_path:
#         return 0
#     if not os.path.isfile(resume_path):
#         raise FileNotFoundError(f"--resume not found: {resume_path}")

#     ckpt = torch.load(resume_path, map_location=device)
#     model.load_state_dict(ckpt["model_state"])
#     lang_clf.load_state_dict(ckpt["lang_clf_state"])
#     opt_model.load_state_dict(ckpt["opt_model_state"])
#     opt_lang.load_state_dict(ckpt["opt_lang_state"])
#     step = int(ckpt.get("step", 0))
#     print(f"[RESUME] loaded {resume_path}, step={step}")
#     return step


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train_jsonl", default="data/groups_train.jsonl")
#     ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
#     ap.add_argument("--run_dir", default="runs/adv4_smoke")
#     ap.add_argument("--resume", default="", help="path to ckpt.pt")

#     ap.add_argument("--backbone", default="google/mt5-small")
#     ap.add_argument("--num_latents", type=int, default=16)
#     ap.add_argument("--latent_dropout", type=float, default=0.1)
#     ap.add_argument("--latent_noise_std", type=float, default=0.01)

#     ap.add_argument("--batch_size", type=int, default=4)
#     ap.add_argument("--grad_accum", type=int, default=8)
#     ap.add_argument("--epochs", type=int, default=1)

#     ap.add_argument("--max_doc_len", type=int, default=256)
#     ap.add_argument("--max_sum_len", type=int, default=64)

#     ap.add_argument("--eval_every", type=int, default=400)
#     ap.add_argument("--max_train_examples", type=int, default=2000)
#     ap.add_argument("--max_valid_examples", type=int, default=200)

#     # core losses
#     ap.add_argument("--lambda_align", type=float, default=0.5)
#     ap.add_argument("--tau", type=float, default=0.07)

#     ap.add_argument("--lambda_varcov", type=float, default=10.0)
#     ap.add_argument("--var_target_std", type=float, default=0.05)

#     # mean controls
#     ap.add_argument("--lambda_mean", type=float, default=0.1)
#     ap.add_argument("--lambda_mean_diff", type=float, default=0.1)

#     # paired closeness
#     ap.add_argument("--lambda_pair", type=float, default=0.2, help="MSE(z_en, z_zh)")

#     # GRL language adversary
#     ap.add_argument("--lambda_lang", type=float, default=1.0, help="weight of language CE with GRL")
#     ap.add_argument("--adv_start_step", type=int, default=300)
#     ap.add_argument("--grl_alpha", type=float, default=1.0, help="GRL strength (gradient multiplier)")
#     ap.add_argument("--grl_warmup", type=int, default=200, help="linear warmup steps after adv_start_step")

#     # optim
#     ap.add_argument("--lr_model", type=float, default=1e-4)
#     ap.add_argument("--lr_lang", type=float, default=1e-3)

#     args = ap.parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     os.makedirs(args.run_dir, exist_ok=True)

#     # 配置保存策略：resume 时不要覆盖原 config.json，避免误导
#     cfg = vars(args)
#     cfg_path = os.path.join(args.run_dir, "config.json")
#     if args.resume and os.path.isfile(cfg_path):
#         with open(os.path.join(args.run_dir, "config_resume.json"), "w", encoding="utf-8") as f:
#             json.dump(cfg, f, indent=2)
#     else:
#         with open(cfg_path, "w", encoding="utf-8") as f:
#             json.dump(cfg, f, indent=2)

#     tok = AutoTokenizer.from_pretrained(args.backbone)
#     train_ds = WikiLinguaGroupDataset(args.train_jsonl, max_examples=(args.max_train_examples or None))
#     valid_ds = WikiLinguaGroupDataset(args.valid_jsonl, max_examples=(args.max_valid_examples or None))
#     collate = make_collate_fn(tok, max_doc_len=args.max_doc_len, max_sum_len=args.max_sum_len)

#     train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
#     valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

#     model = LatentRendererModel(
#         backbone_name=args.backbone,
#         num_latents=args.num_latents,
#         latent_dropout=args.latent_dropout,
#         latent_noise_std=args.latent_noise_std,
#     ).to(device)

#     d_model = model.config.d_model
#     lang_clf = nn.Sequential(
#         nn.Linear(d_model, d_model),
#         nn.ReLU(),
#         nn.Linear(d_model, 2),
#     ).to(device)

#     opt_model = torch.optim.AdamW(model.parameters(), lr=args.lr_model)
#     opt_lang = torch.optim.AdamW(lang_clf.parameters(), lr=args.lr_lang)

#     # resume
#     step = try_load_checkpoint(args.resume, model, lang_clf, opt_model, opt_lang, device)

#     t0 = time.time()
#     log_path = os.path.join(args.run_dir, "logs.jsonl")
#     ckpt_path = os.path.join(args.run_dir, "ckpt.pt")

#     # 注意：resume 训练不强求完全复现随机性，这里仍固定 seed
#     random.seed(42)
#     torch.manual_seed(42)

#     opt_model.zero_grad(set_to_none=True)
#     opt_lang.zero_grad(set_to_none=True)

#     for epoch in range(args.epochs):
#         model.train()
#         lang_clf.train()

#         for batch in train_dl:
#             step += 1
#             en_ids = batch["en_input_ids"].to(device)
#             en_m = batch["en_attention_mask"].to(device)
#             zh_ids = batch["zh_input_ids"].to(device)
#             zh_m = batch["zh_attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             out = model(en_ids, en_m, zh_ids, zh_m, labels)
#             z_en = out.zbar_en
#             z_zh = out.zbar_zh
#             z_all = torch.cat([z_en, z_zh], dim=0)

#             y_lang = torch.cat(
#                 [
#                     torch.zeros(z_en.size(0), dtype=torch.long, device=device),
#                     torch.ones(z_zh.size(0), dtype=torch.long, device=device),
#                 ],
#                 dim=0,
#             )

#             loss_en = ce_loss_from_logits(out.logits_en, labels)
#             loss_zh = ce_loss_from_logits(out.logits_zh, labels)
#             loss_render = 0.5 * (loss_en + loss_zh)

#             loss_align = info_nce_loss(z_en, z_zh, tau=args.tau) if args.lambda_align > 0 else torch.tensor(0.0, device=device)

#             loss_varcov = torch.tensor(0.0, device=device)
#             if args.lambda_varcov > 0:
#                 loss_varcov = varcov_regularizer(z_en, args.var_target_std) + varcov_regularizer(z_zh, args.var_target_std)

#             mu_all = z_all.mean(dim=0)
#             mu_en = z_en.mean(dim=0)
#             mu_zh = z_zh.mean(dim=0)
#             loss_mean = (mu_all ** 2).sum()
#             loss_mean_diff = ((mu_en - mu_zh) ** 2).sum()

#             loss_pair = ((z_en - z_zh) ** 2).mean()

#             # GRL adversary
#             loss_lang = torch.tensor(0.0, device=device)
#             acc_lang = 0.0
#             alpha = 0.0
#             if step >= args.adv_start_step and args.lambda_lang > 0:
#                 k = min(max(step - args.adv_start_step, 0), args.grl_warmup)
#                 alpha = args.grl_alpha * (k / max(args.grl_warmup, 1))
#                 z_rev = grad_reverse(z_all, alpha)
#                 logits_lang = lang_clf(z_rev)
#                 loss_lang = F.cross_entropy(logits_lang, y_lang)
#                 with torch.no_grad():
#                     acc_lang = (logits_lang.argmax(dim=1) == y_lang).float().mean().item()

#             loss = (
#                 loss_render
#                 + args.lambda_align * loss_align
#                 + args.lambda_varcov * loss_varcov
#                 + args.lambda_mean * loss_mean
#                 + args.lambda_mean_diff * loss_mean_diff
#                 + args.lambda_pair * loss_pair
#                 + args.lambda_lang * loss_lang
#             )

#             (loss / args.grad_accum).backward()

#             if step % args.grad_accum == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 opt_model.step()
#                 opt_lang.step()
#                 opt_model.zero_grad(set_to_none=True)
#                 opt_lang.zero_grad(set_to_none=True)

#             if step % 20 == 0:
#                 # 额外打一个 mean_diff 的 L2 norm，便于解释 probe 为什么还能接近 1.0
#                 with torch.no_grad():
#                     mean_diff_l2 = float((mu_en - mu_zh).pow(2).sum().sqrt().item())
#                 msg = {
#                     "step": step,
#                     "epoch": epoch,
#                     "loss": float(loss.item()),
#                     "loss_render": float(loss_render.item()),
#                     "loss_align": float(loss_align.item()),
#                     "loss_varcov": float(loss_varcov.item()),
#                     "loss_mean": float(loss_mean.item()),
#                     "loss_mean_diff": float(loss_mean_diff.item()),
#                     "mean_diff_l2": mean_diff_l2,
#                     "loss_pair": float(loss_pair.item()),
#                     "loss_lang": float(loss_lang.item()),
#                     "grl_alpha_eff": float(alpha),
#                     "lang_acc_batch": float(acc_lang),
#                     "elapsed_sec": float(time.time() - t0),
#                 }
#                 print(msg)
#                 with open(log_path, "a", encoding="utf-8") as f:
#                     f.write(json.dumps(msg) + "\n")

#             if step % args.eval_every == 0:
#                 diag = {"step": step, **eval_nll(model, valid_dl, device)}
#                 print("EVAL:", diag)
#                 with open(os.path.join(args.run_dir, "diag.jsonl"), "a", encoding="utf-8") as f:
#                     f.write(json.dumps(diag) + "\n")
#                 save_checkpoint(ckpt_path, model, lang_clf, opt_model, opt_lang, step, cfg)

#     save_checkpoint(ckpt_path, model, lang_clf, opt_model, opt_lang, step, cfg)
#     print(f"Done. Saved checkpoint to {ckpt_path}")


# if __name__ == "__main__":
#     main()

from __future__ import annotations

import argparse
import json
import os
import time
import random
from typing import Dict, Any, Optional

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


class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradReverseFn.apply(x, alpha)


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


def try_load_checkpoint(
    resume_path: str,
    model: nn.Module,
    lang_clf: nn.Module,
    opt_model: torch.optim.Optimizer,
    opt_lang: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not resume_path:
        return 0
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"--resume not found: {resume_path}")

    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    lang_clf.load_state_dict(ckpt["lang_clf_state"])
    opt_model.load_state_dict(ckpt["opt_model_state"])
    opt_lang.load_state_dict(ckpt["opt_lang_state"])
    step = int(ckpt.get("step", 0))
    print(f"[RESUME] loaded {resume_path}, step={step}")
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
    ap.add_argument("--run_dir", default="runs/adv4_smoke")
    ap.add_argument("--resume", default="", help="path to ckpt.pt")

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

    # core losses
    ap.add_argument("--lambda_align", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.07)

    ap.add_argument("--lambda_varcov", type=float, default=10.0)
    ap.add_argument("--var_target_std", type=float, default=0.05)

    # mean controls
    ap.add_argument("--lambda_mean", type=float, default=0.1)
    ap.add_argument("--lambda_mean_diff", type=float, default=0.1)

    # paired closeness
    ap.add_argument("--lambda_pair", type=float, default=0.2, help="MSE(z_en, z_zh)")

    # GRL language adversary
    ap.add_argument("--lambda_lang", type=float, default=1.0, help="weight of language CE with GRL")
    ap.add_argument("--adv_start_step", type=int, default=300)
    ap.add_argument("--grl_alpha", type=float, default=1.0, help="GRL strength (gradient multiplier)")
    ap.add_argument("--grl_warmup", type=int, default=200, help="linear warmup steps after adv_start_step")

    # optim
    ap.add_argument("--lr_model", type=float, default=1e-4)
    ap.add_argument("--lr_lang", type=float, default=1e-3)

    # added
    ap.add_argument("--adv_clf_steps", type=int, default=1, help="lang_clf updates per batch (encoder frozen)")
    ap.add_argument("--adv_clf_weight_decay", type=float, default=0.0)  # 可选

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.run_dir, exist_ok=True)

    # 配置保存策略：resume 时不要覆盖原 config.json，避免误导
    cfg = vars(args)
    cfg_path = os.path.join(args.run_dir, "config.json")
    if args.resume and os.path.isfile(cfg_path):
        with open(os.path.join(args.run_dir, "config_resume.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    else:
        with open(cfg_path, "w", encoding="utf-8") as f:
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
    lang_clf = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, 2),
    ).to(device)

    opt_model = torch.optim.AdamW(model.parameters(), lr=args.lr_model)
    opt_lang = torch.optim.AdamW(lang_clf.parameters(), lr=args.lr_lang, weight_decay=args.adv_clf_weight_decay)

    # resume
    step = try_load_checkpoint(args.resume, model, lang_clf, opt_model, opt_lang, device)

    t0 = time.time()
    log_path = os.path.join(args.run_dir, "logs.jsonl")
    ckpt_path = os.path.join(args.run_dir, "ckpt.pt")

    # 注意：resume 训练不强求完全复现随机性，这里仍固定 seed
    random.seed(42)
    torch.manual_seed(42)

    opt_model.zero_grad(set_to_none=True)
    opt_lang.zero_grad(set_to_none=True)

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
            z_en = out.zbar_en
            z_zh = out.zbar_zh
            z_all = torch.cat([z_en, z_zh], dim=0)

            y_lang = torch.cat(
                [
                    torch.zeros(z_en.size(0), dtype=torch.long, device=device),
                    torch.ones(z_zh.size(0), dtype=torch.long, device=device),
                ],
                dim=0,
            )

            loss_en = ce_loss_from_logits(out.logits_en, labels)
            loss_zh = ce_loss_from_logits(out.logits_zh, labels)
            loss_render = 0.5 * (loss_en + loss_zh)

            loss_align = info_nce_loss(z_en, z_zh, tau=args.tau) if args.lambda_align > 0 else torch.tensor(0.0, device=device)

            loss_varcov = torch.tensor(0.0, device=device)
            if args.lambda_varcov > 0:
                loss_varcov = varcov_regularizer(z_en, args.var_target_std) + varcov_regularizer(z_zh, args.var_target_std)

            mu_all = z_all.mean(dim=0)
            mu_en = z_en.mean(dim=0)
            mu_zh = z_zh.mean(dim=0)
            loss_mean = (mu_all ** 2).sum()
            loss_mean_diff = ((mu_en - mu_zh) ** 2).sum()

            loss_pair = ((z_en - z_zh) ** 2).mean()

            # GRL adversary
            # loss_lang = torch.tensor(0.0, device=device)
            # acc_lang = 0.0
            # alpha = 0.0
            # if step >= args.adv_start_step and args.lambda_lang > 0:
            #     k = min(max(step - args.adv_start_step, 0), args.grl_warmup)
            #     alpha = args.grl_alpha * (k / max(args.grl_warmup, 1))
            #     z_rev = grad_reverse(z_all, alpha)
            #     logits_lang = lang_clf(z_rev)
            #     loss_lang = F.cross_entropy(logits_lang, y_lang)
            #     with torch.no_grad():
            #         acc_lang = (logits_lang.argmax(dim=1) == y_lang).float().mean().item()
            def set_requires_grad(m: nn.Module, flag: bool) -> None:
                for p in m.parameters():
                    p.requires_grad_(flag)
            
            loss_lang = torch.tensor(0.0, device=device)
            acc_lang_batch = 0.0                # encoder-adversarial view
            acc_lang_clf_detached = 0.0         # clf-only view
            alpha = 0.0
            
            adv_active = (step >= args.adv_start_step) and (args.lambda_lang > 0)
            
            if adv_active:
                # 1) Train lang_clf only on detached features (make clf strong)
                set_requires_grad(lang_clf, True)
                for _ in range(max(args.adv_clf_steps, 1)):
                    opt_lang.zero_grad(set_to_none=True)
                    logits_det = lang_clf(z_all.detach())
                    loss_clf = F.cross_entropy(logits_det, y_lang)
                    loss_clf.backward()
                    opt_lang.step()
                    with torch.no_grad():
                        acc_lang_clf_detached = (logits_det.argmax(dim=1) == y_lang).float().mean().item()
            
                # 2) Train encoder to confuse a fixed (strong) clf via GRL
                k = min(max(step - args.adv_start_step, 0), args.grl_warmup)
                alpha = args.grl_alpha * (k / max(args.grl_warmup, 1))
            
                set_requires_grad(lang_clf, False)  # freeze clf weights
                z_rev = grad_reverse(z_all, alpha)
                logits_adv = lang_clf(z_rev)        # gradients flow to z_rev, not to clf params
                loss_lang = F.cross_entropy(logits_adv, y_lang)
                with torch.no_grad():
                    acc_lang_batch = (logits_adv.argmax(dim=1) == y_lang).float().mean().item()
                set_requires_grad(lang_clf, True)

            loss = (
                loss_render
                + args.lambda_align * loss_align
                + args.lambda_varcov * loss_varcov
                + args.lambda_mean * loss_mean
                + args.lambda_mean_diff * loss_mean_diff
                + args.lambda_pair * loss_pair
                + args.lambda_lang * loss_lang
            )

            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_model.step()
                # opt_lang.step()
                opt_model.zero_grad(set_to_none=True)
                # opt_lang.zero_grad(set_to_none=True)

            if step % 20 == 0:
                # 额外打一个 mean_diff 的 L2 norm，便于解释 probe 为什么还能接近 1.0
                with torch.no_grad():
                    mean_diff_l2 = float((mu_en - mu_zh).pow(2).sum().sqrt().item())
                msg = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "loss_render": float(loss_render.item()),
                    "loss_align": float(loss_align.item()),
                    "loss_varcov": float(loss_varcov.item()),
                    "loss_mean": float(loss_mean.item()),
                    "loss_mean_diff": float(loss_mean_diff.item()),
                    "mean_diff_l2": mean_diff_l2,
                    "loss_pair": float(loss_pair.item()),
                    "loss_lang": float(loss_lang.item()),
                    "grl_alpha_eff": float(alpha),
                    # "lang_acc_batch": float(acc_lang),
                    "lang_acc_batch": float(acc_lang_batch),
                    "lang_acc_clf_detached": float(acc_lang_clf_detached),
                    "grl_alpha_eff": float(alpha),
                    # 
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
