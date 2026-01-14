# # # # src/train_planB_ramp_adv.py
# # # import os
# # # import shlex
# # # import subprocess
# # # import sys
# # # from dataclasses import dataclass


# # # @dataclass
# # # class Flags:
# # #     TRAIN_ENTRY: str = os.environ.get("TRAIN_ENTRY", "src/train_adv.py")

# # #     RUN_DIR_FLAG: str = os.environ.get("RUN_DIR_FLAG", "--run_dir")
# # #     RESUME_FLAG: str = os.environ.get("RESUME_FLAG", "--resume")  # 改成你实际用的，如 --resume_ckpt
# # #     SEED_FLAG: str = os.environ.get("SEED_FLAG", "--seed")
# # #     BATCH_SIZE_FLAG: str = os.environ.get("BATCH_SIZE_FLAG", "--batch_size")
# # #     MAX_STEPS_FLAG: str = os.environ.get("MAX_STEPS_FLAG", "--max_steps")
# # #     LR_FLAG: str = os.environ.get("LR_FLAG", "--lr")
# # #     WD_FLAG: str = os.environ.get("WD_FLAG", "--weight_decay")
# # #     GRAD_CLIP_FLAG: str = os.environ.get("GRAD_CLIP_FLAG", "--grad_clip")

# # #     L_ALIGN_FLAG: str = os.environ.get("L_ALIGN_FLAG", "--lambda_align")
# # #     L_VARCOV_FLAG: str = os.environ.get("L_VARCOV_FLAG", "--lambda_varcov")
# # #     L_MEAN_FLAG: str = os.environ.get("L_MEAN_FLAG", "--lambda_mean")
# # #     L_MEAN_DIFF_FLAG: str = os.environ.get("L_MEAN_DIFF_FLAG", "--lambda_mean_diff")
# # #     L_PAIR_FLAG: str = os.environ.get("L_PAIR_FLAG", "--lambda_pair")
# # #     L_LANG_FLAG: str = os.environ.get("L_LANG_FLAG", "--lambda_lang")

# # #     GRL_ALPHA_FLAG: str = os.environ.get("GRL_ALPHA_FLAG", "--grl_alpha")
# # #     LANG_CLF_LR_FLAG: str = os.environ.get("LANG_CLF_LR_FLAG", "--lang_clf_lr")
# # #     LANG_CLF_STEPS_FLAG: str = os.environ.get("LANG_CLF_STEPS_FLAG", "--lang_clf_steps")


# # # def run(cmd: list[str]) -> None:
# # #     print("\n[CMD]")
# # #     print(" ".join(shlex.quote(x) for x in cmd))
# # #     subprocess.run(cmd, check=True)


# # # def stage_cmd(flags: Flags, *, run_dir: str, seed: int, batch_size: int, max_steps: int,
# # #               lr: float, wd: float, grad_clip: float,
# # #               lambda_align: float, lambda_varcov: float, lambda_mean: float, lambda_mean_diff: float, lambda_pair: float,
# # #               lambda_lang: float, grl_alpha: float, lang_clf_lr: float, lang_clf_steps: int,
# # #               resume_ckpt: str | None) -> list[str]:
# # #     cmd = [
# # #         sys.executable, flags.TRAIN_ENTRY,
# # #         flags.RUN_DIR_FLAG, run_dir,
# # #         flags.SEED_FLAG, str(seed),
# # #         flags.BATCH_SIZE_FLAG, str(batch_size),
# # #         flags.MAX_STEPS_FLAG, str(max_steps),
# # #         flags.LR_FLAG, str(lr),
# # #         flags.WD_FLAG, str(wd),
# # #         flags.GRAD_CLIP_FLAG, str(grad_clip),

# # #         flags.L_ALIGN_FLAG, str(lambda_align),
# # #         flags.L_VARCOV_FLAG, str(lambda_varcov),
# # #         flags.L_MEAN_FLAG, str(lambda_mean),
# # #         flags.L_MEAN_DIFF_FLAG, str(lambda_mean_diff),
# # #         flags.L_PAIR_FLAG, str(lambda_pair),

# # #         flags.L_LANG_FLAG, str(lambda_lang),
# # #         flags.GRL_ALPHA_FLAG, str(grl_alpha),
# # #         flags.LANG_CLF_LR_FLAG, str(lang_clf_lr),
# # #         flags.LANG_CLF_STEPS_FLAG, str(lang_clf_steps),
# # #     ]
# # #     if resume_ckpt:
# # #         cmd += [flags.RESUME_FLAG, resume_ckpt]
# # #     return cmd


# # # def main():
# # #     f = Flags()

# # #     run_dir = os.environ.get("RUN_DIR", "runs/planB_ramp_adv")
# # #     seed = int(os.environ.get("SEED", "0"))
# # #     batch_size = int(os.environ.get("BATCH_SIZE", "8"))

# # #     # 三阶段步数（都是 “max_steps=累计步数” 的风格）
# # #     warmup_steps = int(os.environ.get("WARMUP_STEPS", "2000"))
# # #     ramp_steps = int(os.environ.get("RAMP_STEPS", "4000"))
# # #     final_steps = int(os.environ.get("FINAL_STEPS", "20000"))

# # #     lr = float(os.environ.get("LR", "3e-4"))
# # #     wd = float(os.environ.get("WD", "0.01"))
# # #     grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))

# # #     # 共同正则项，B 方案倾向温和一些，先保 inv / nll
# # #     lambda_align = float(os.environ.get("LAMBDA_ALIGN", "1.0"))
# # #     lambda_varcov = float(os.environ.get("LAMBDA_VARCOV", "0.1"))
# # #     lambda_mean = float(os.environ.get("LAMBDA_MEAN", "0.5"))
# # #     lambda_mean_diff = float(os.environ.get("LAMBDA_MEAN_DIFF", "0.1"))
# # #     lambda_pair = float(os.environ.get("LAMBDA_PAIR", "0.005"))

# # #     lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "5e-4"))
# # #     lang_clf_steps = int(os.environ.get("LANG_CLF_STEPS", "1"))

# # #     ckpt_path = f"{run_dir}/ckpt.pt"

# # #     # Stage 1: warmup, 不做语言对抗，让基本任务先收敛一点
# # #     cmd1 = stage_cmd(
# # #         f, run_dir=run_dir, seed=seed, batch_size=batch_size, max_steps=warmup_steps,
# # #         lr=lr, wd=wd, grad_clip=grad_clip,
# # #         lambda_align=lambda_align, lambda_varcov=lambda_varcov,
# # #         lambda_mean=0.0, lambda_mean_diff=0.0, lambda_pair=0.0,
# # #         lambda_lang=0.0, grl_alpha=0.0,
# # #         lang_clf_lr=lang_clf_lr, lang_clf_steps=lang_clf_steps,
# # #         resume_ckpt=None
# # #     )
# # #     run(cmd1)

# # #     # Stage 2: ramp, 逐渐加对抗但不拉满
# # #     # 这里用中等强度，目标是把 probe 先从 ~0.996 拉到更低，同时 inv 不要崩
# # #     cmd2 = stage_cmd(
# # #         f, run_dir=run_dir, seed=seed, batch_size=batch_size, max_steps=ramp_steps,
# # #         lr=lr, wd=wd, grad_clip=grad_clip,
# # #         lambda_align=lambda_align, lambda_varcov=lambda_varcov,
# # #         lambda_mean=lambda_mean, lambda_mean_diff=lambda_mean_diff, lambda_pair=lambda_pair,
# # #         lambda_lang=float(os.environ.get("LAMBDA_LANG_RAMP", "1.0")),
# # #         grl_alpha=float(os.environ.get("GRL_ALPHA_RAMP", "1.0")),
# # #         lang_clf_lr=lang_clf_lr, lang_clf_steps=lang_clf_steps,
# # #         resume_ckpt=ckpt_path
# # #     )
# # #     run(cmd2)

# # #     # Stage 3: final, 稳态对抗，略增强
# # #     cmd3 = stage_cmd(
# # #         f, run_dir=run_dir, seed=seed, batch_size=batch_size, max_steps=final_steps,
# # #         lr=lr, wd=wd, grad_clip=grad_clip,
# # #         lambda_align=lambda_align, lambda_varcov=lambda_varcov,
# # #         lambda_mean=lambda_mean, lambda_mean_diff=lambda_mean_diff, lambda_pair=lambda_pair,
# # #         lambda_lang=float(os.environ.get("LAMBDA_LANG_FINAL", "2.0")),
# # #         grl_alpha=float(os.environ.get("GRL_ALPHA_FINAL", "2.0")),
# # #         lang_clf_lr=lang_clf_lr, lang_clf_steps=lang_clf_steps,
# # #         resume_ckpt=ckpt_path
# # #     )
# # #     run(cmd3)

# # #     print("\n[Next]")
# # #     print(f"python src/eval_diag.py --ckpt {ckpt_path} --batch_size 8")
# # #     print(f"python src/probe_leakage.py --ckpt {ckpt_path} --max_train_groups 8000 --max_valid_groups 946")


# # # if __name__ == "__main__":
# # #     main()

# # # # src/train_planB_ramp_adv.py
# # # import os
# # # import shlex
# # # import subprocess
# # # import sys
# # # from dataclasses import dataclass

# # # @dataclass
# # # class Flags:
# # #     TRAIN_ENTRY: str = os.environ.get("TRAIN_ENTRY", "src/train_adv4.py")

# # #     RUN_DIR_FLAG: str = os.environ.get("RUN_DIR_FLAG", "--run_dir")
# # #     # Note: train_adv4.py currently doesn't support --resume. 
# # #     # Commenting out to avoid "unrecognized argument" error.
# # #     # RESUME_FLAG: str = os.environ.get("RESUME_FLAG", "--resume") 
    
# # #     BATCH_SIZE_FLAG: str = os.environ.get("BATCH_SIZE_FLAG", "--batch_size")
# # #     EPOCHS_FLAG: str = os.environ.get("EPOCHS_FLAG", "--epochs") # train_adv4 uses epochs
    
# # #     LR_FLAG: str = os.environ.get("LR_FLAG", "--lr_model") # Fixed ambiguity
    
# # #     L_ALIGN_FLAG: str = os.environ.get("L_ALIGN_FLAG", "--lambda_align")
# # #     L_VARCOV_FLAG: str = os.environ.get("L_VARCOV_FLAG", "--lambda_varcov")
# # #     L_MEAN_FLAG: str = os.environ.get("L_MEAN_FLAG", "--lambda_mean")
# # #     L_MEAN_DIFF_FLAG: str = os.environ.get("L_MEAN_DIFF_FLAG", "--lambda_mean_diff")
# # #     L_PAIR_FLAG: str = os.environ.get("L_PAIR_FLAG", "--lambda_pair")
# # #     L_LANG_FLAG: str = os.environ.get("L_LANG_FLAG", "--lambda_lang")

# # #     GRL_ALPHA_FLAG: str = os.environ.get("GRL_ALPHA_FLAG", "--grl_alpha")
# # #     LANG_CLF_LR_FLAG: str = os.environ.get("LANG_CLF_LR_FLAG", "--lr_lang") # Fixed name

# # # def run(cmd: list[str]) -> None:
# # #     print("\n[CMD]")
# # #     print(" ".join(shlex.quote(x) for x in cmd))
# # #     subprocess.run(cmd, check=True)

# # # def stage_cmd(flags: Flags, *, run_dir: str, batch_size: int, epochs: int,
# # #               lr: float, lambda_align: float, lambda_varcov: float, 
# # #               lambda_mean: float, lambda_mean_diff: float, lambda_pair: float,
# # #               lambda_lang: float, grl_alpha: float, lang_clf_lr: float) -> list[str]:
# # #     cmd = [
# # #         sys.executable, flags.TRAIN_ENTRY,
# # #         flags.RUN_DIR_FLAG, run_dir,
# # #         flags.BATCH_SIZE_FLAG, str(batch_size),
# # #         flags.EPOCHS_FLAG, str(epochs),
# # #         flags.LR_FLAG, str(lr),
# # #         flags.L_ALIGN_FLAG, str(lambda_align),
# # #         flags.L_VARCOV_FLAG, str(lambda_varcov),
# # #         flags.L_MEAN_FLAG, str(lambda_mean),
# # #         flags.L_MEAN_DIFF_FLAG, str(lambda_mean_diff),
# # #         flags.L_PAIR_FLAG, str(lambda_pair),
# # #         flags.L_LANG_FLAG, str(lambda_lang),
# # #         flags.GRL_ALPHA_FLAG, str(grl_alpha),
# # #         flags.LANG_CLF_LR_FLAG, str(lang_clf_lr),
# # #     ]
# # #     return cmd

# # # def main():
# # #     f = Flags()

# # #     run_dir = os.environ.get("RUN_DIR", "runs/planB_ramp_adv")
# # #     batch_size = int(os.environ.get("BATCH_SIZE", "8"))

# # #     # train_adv4 uses epochs. Setting small epoch counts for stages 
# # #     # as proxy for steps if you aren't modifying train_adv4.py
# # #     warmup_epochs = 1 
# # #     ramp_epochs = 2
# # #     final_epochs = 5

# # #     lr = float(os.environ.get("LR", "3e-4"))
# # #     lambda_align = float(os.environ.get("LAMBDA_ALIGN", "1.0"))
# # #     lambda_varcov = float(os.environ.get("LAMBDA_VARCOV", "0.1"))
# # #     lambda_mean = float(os.environ.get("LAMBDA_MEAN", "0.5"))
# # #     lambda_mean_diff = float(os.environ.get("LAMBDA_MEAN_DIFF", "0.1"))
# # #     lambda_pair = float(os.environ.get("LAMBDA_PAIR", "0.005"))
# # #     lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "5e-4"))

# # #     ckpt_path = f"{run_dir}/ckpt.pt"

# # #     # Stage 1: warmup
# # #     cmd1 = stage_cmd(
# # #         f, run_dir=run_dir, batch_size=batch_size, epochs=warmup_epochs,
# # #         lr=lr, lambda_align=lambda_align, lambda_varcov=lambda_varcov,
# # #         lambda_mean=0.0, lambda_mean_diff=0.0, lambda_pair=0.0,
# # #         lambda_lang=0.0, grl_alpha=0.0, lang_clf_lr=lang_clf_lr
# # #     )
# # #     run(cmd1)

# # #     # Stage 2: ramp
# # #     cmd2 = stage_cmd(
# # #         f, run_dir=run_dir, batch_size=batch_size, epochs=ramp_epochs,
# # #         lr=lr, lambda_align=lambda_align, lambda_varcov=lambda_varcov,
# # #         lambda_mean=lambda_mean, lambda_mean_diff=lambda_mean_diff, lambda_pair=lambda_pair,
# # #         lambda_lang=float(os.environ.get("LAMBDA_LANG_RAMP", "1.0")),
# # #         grl_alpha=float(os.environ.get("GRL_ALPHA_RAMP", "1.0")),
# # #         lang_clf_lr=lang_clf_lr
# # #     )
# # #     run(cmd2)

# # #     # Stage 3: final
# # #     cmd3 = stage_cmd(
# # #         f, run_dir=run_dir, batch_size=batch_size, epochs=final_epochs,
# # #         lr=lr, lambda_align=lambda_align, lambda_varcov=lambda_varcov,
# # #         lambda_mean=lambda_mean, lambda_mean_diff=lambda_mean_diff, lambda_pair=lambda_pair,
# # #         lambda_lang=float(os.environ.get("LAMBDA_LANG_FINAL", "2.0")),
# # #         grl_alpha=float(os.environ.get("GRL_ALPHA_FINAL", "2.0")),
# # #         lang_clf_lr=lang_clf_lr
# # #     )
# # #     run(cmd3)

# # #     print("\n[Next]")
# # #     print(f"python src/eval_diag.py --ckpt {ckpt_path} --batch_size 8")
# # #     print(f"python src/probe_leakage.py --ckpt {run_dir}/ckpt.pt --max_train_groups 8000 --max_valid_groups 946")

# # # if __name__ == "__main__":
# # #     main()


# # # src/train_planB_ramp_adv.py
# # import os
# # import shlex
# # import subprocess
# # import sys
# # from dataclasses import dataclass

# # @dataclass
# # class Flags:
# #     TRAIN_ENTRY: str = os.environ.get("TRAIN_ENTRY", "src/train_adv4.py")

# #     RUN_DIR_FLAG: str = "--run_dir"
# #     RESUME_FLAG: str = "--resume"

# #     BATCH_SIZE_FLAG: str = "--batch_size"
# #     EPOCHS_FLAG: str = "--epochs"
# #     LR_FLAG: str = "--lr_model"

# #     L_ALIGN_FLAG: str = "--lambda_align"
# #     L_VARCOV_FLAG: str = "--lambda_varcov"
# #     L_MEAN_FLAG: str = "--lambda_mean"
# #     L_MEAN_DIFF_FLAG: str = "--lambda_mean_diff"
# #     L_PAIR_FLAG: str = "--lambda_pair"
# #     L_LANG_FLAG: str = "--lambda_lang"

# #     ADV_START_FLAG: str = "--adv_start_step"
# #     GRL_WARMUP_FLAG: str = "--grl_warmup"
# #     GRL_ALPHA_FLAG: str = "--grl_alpha"
# #     LANG_CLF_LR_FLAG: str = "--lr_lang"

# #     # added
# #     ADV_CLF_STEPS_FLAG: str = "--adv_clf_steps"


# # def run(cmd: list[str]) -> None:
# #     print("\n[CMD]")
# #     print(" ".join(shlex.quote(x) for x in cmd))
# #     subprocess.run(cmd, check=True)


# # def stage_cmd(
# #     flags: Flags,
# #     *,
# #     run_dir: str,
# #     resume: str,
# #     batch_size: int,
# #     epochs: int,
# #     lr: float,
# #     lambda_align: float,
# #     lambda_varcov: float,
# #     lambda_mean: float,
# #     lambda_mean_diff: float,
# #     lambda_pair: float,
# #     lambda_lang: float,
# #     adv_start_step: int,
# #     grl_warmup: int,
# #     grl_alpha: float,
# #     lang_clf_lr: float,
# #     adv_clf_steps: int,
# # ) -> list[str]:
# #     cmd = [
# #         sys.executable,
# #         flags.TRAIN_ENTRY,
# #         flags.RUN_DIR_FLAG, run_dir,
# #         flags.BATCH_SIZE_FLAG, str(batch_size),
# #         flags.EPOCHS_FLAG, str(epochs),
# #         flags.LR_FLAG, str(lr),

# #         flags.L_ALIGN_FLAG, str(lambda_align),
# #         flags.L_VARCOV_FLAG, str(lambda_varcov),
# #         flags.L_MEAN_FLAG, str(lambda_mean),
# #         flags.L_MEAN_DIFF_FLAG, str(lambda_mean_diff),
# #         flags.L_PAIR_FLAG, str(lambda_pair),

# #         flags.L_LANG_FLAG, str(lambda_lang),
# #         flags.ADV_START_FLAG, str(adv_start_step),
# #         flags.GRL_WARMUP_FLAG, str(grl_warmup),
# #         flags.GRL_ALPHA_FLAG, str(grl_alpha),
# #         flags.LANG_CLF_LR_FLAG, str(lang_clf_lr),
# #         # added
# #         flags.ADV_CLF_STEPS_FLAG, str(adv_clf_steps),
# #     ]
# #     if resume:
# #         cmd += [flags.RESUME_FLAG, resume]
# #     return cmd


# # def main():
# #     f = Flags()

# #     run_dir = os.environ.get("RUN_DIR", "runs/planB_ramp_adv")
# #     batch_size = int(os.environ.get("BATCH_SIZE", "8"))
# #     lr = float(os.environ.get("LR", "3e-4"))

# #     lambda_align = float(os.environ.get("LAMBDA_ALIGN", "1.0"))
# #     lambda_varcov = float(os.environ.get("LAMBDA_VARCOV", "0.1"))
# #     lambda_mean = float(os.environ.get("LAMBDA_MEAN", "0.5"))
# #     lambda_mean_diff = float(os.environ.get("LAMBDA_MEAN_DIFF", "0.1"))
# #     # lambda_pair = float(os.environ.get("LAMBDA_PAIR", "0.005"))
# #     lambda_pair = float(os.environ.get("LAMBDA_PAIR", "0.05"))
# #     # lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "5e-4"))
# #     lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "0.003"))
# #     # adv_clf_steps = float(os.environ.get("ADV_CLF_STEPS_FLAG", "8"))

# #     ckpt_path = f"{run_dir}/ckpt.pt"

# #     # 你原来 1/2/5 个 epoch，这里保留
# #     warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", "1"))
# #     ramp_epochs = int(os.environ.get("RAMP_EPOCHS", "2"))
# #     final_epochs = int(os.environ.get("FINAL_EPOCHS", "5"))

# #     # Stage 1: 预热 lang_clf（alpha=0，不反传 encoder），立即开始训练 clf
# #     cmd1 = stage_cmd(
# #         f,
# #         run_dir=run_dir,
# #         resume="",
# #         batch_size=batch_size,
# #         epochs=warmup_epochs,
# #         lr=lr,
# #         lambda_align=lambda_align,
# #         lambda_varcov=lambda_varcov,
# #         lambda_mean=0.0,
# #         lambda_mean_diff=0.0,
# #         lambda_pair=0.0,
# #         lambda_lang=float(os.environ.get("LAMBDA_LANG_WARMUP", "1.0")),
# #         adv_start_step=0,
# #         grl_warmup=int(os.environ.get("GRL_WARMUP", "200")),
# #         grl_alpha=0.0,
# #         lang_clf_lr=lang_clf_lr,
# #         # added
# #         adv_clf_steps=16,
# #     )
# #     run(cmd1)

# #     # Stage 2: ramp（resume），开始把 alpha 拉起来
# #     cmd2 = stage_cmd(
# #         f,
# #         run_dir=run_dir,
# #         resume=ckpt_path,
# #         batch_size=batch_size,
# #         epochs=ramp_epochs,
# #         lr=lr,
# #         lambda_align=lambda_align,
# #         lambda_varcov=lambda_varcov,
# #         lambda_mean=lambda_mean,
# #         lambda_mean_diff=lambda_mean_diff,
# #         lambda_pair=lambda_pair,
# #         lambda_lang=float(os.environ.get("LAMBDA_LANG_RAMP", "1.0")),
# #         adv_start_step=0,
# #         grl_warmup=int(os.environ.get("GRL_WARMUP", "200")),
# #         grl_alpha=float(os.environ.get("GRL_ALPHA_RAMP", "1.0")),
# #         lang_clf_lr=lang_clf_lr,
# #         # added
# #         adv_clf_steps=12,
# #     )
# #     run(cmd2)

# #     # Stage 3: final（resume）
# #     cmd3 = stage_cmd(
# #         f,
# #         run_dir=run_dir,
# #         resume=ckpt_path,
# #         batch_size=batch_size,
# #         epochs=final_epochs,
# #         lr=lr,
# #         lambda_align=lambda_align,
# #         lambda_varcov=lambda_varcov,
# #         lambda_mean=lambda_mean,
# #         lambda_mean_diff=lambda_mean_diff,
# #         lambda_pair=lambda_pair,
# #         lambda_lang=float(os.environ.get("LAMBDA_LANG_FINAL", "2.0")),
# #         adv_start_step=0,
# #         grl_warmup=int(os.environ.get("GRL_WARMUP", "200")),
# #         grl_alpha=float(os.environ.get("GRL_ALPHA_FINAL", "2.0")),
# #         lang_clf_lr=lang_clf_lr,
# #         # added
# #         adv_clf_steps=8,
# #     )
# #     run(cmd3)

# #     print("\n[Next]")
# #     print(f"python src/eval_diag.py --ckpt {ckpt_path} --batch_size 8")
# #     print(f"python src/probe_leakage.py --ckpt {ckpt_path} --max_train_groups 8000 --max_valid_groups 946")


# # if __name__ == "__main__":
# #     main()


# from __future__ import annotations

# import argparse
# import json
# import os
# import time
# import random
# from typing import Dict, Any, Optional, Tuple

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


# def set_requires_grad(m: nn.Module, flag: bool) -> None:
#     for p in m.parameters():
#         p.requires_grad_(flag)


# def length_bucket_from_mask(attn_mask: torch.Tensor) -> torch.Tensor:
#     """
#     attn_mask: [B, T] (1 keep, 0 pad)
#     buckets:
#       0: 1-32
#       1: 33-64
#       2: 65-128
#       3: 129+
#     """
#     lengths = attn_mask.sum(dim=1).long()
#     b = torch.zeros_like(lengths)
#     b = b + (lengths > 32).long()
#     b = b + (lengths > 64).long()
#     b = b + (lengths > 128).long()
#     return b.clamp(0, 3)


# class AdvFeatureBuffer:
#     """
#     CPU ring buffer for adversary training.
#     Stores:
#       feat: [Q, D]
#       y_lang: [Q] (0 en, 1 zh)
#       y_len:  [Q] (0..3)
#     """

#     def __init__(self, size: int, dim: int) -> None:
#         self.size = int(size)
#         self.dim = int(dim)
#         self.feat = torch.empty((self.size, self.dim), dtype=torch.float32)
#         self.y_lang = torch.empty((self.size,), dtype=torch.long)
#         self.y_len = torch.empty((self.size,), dtype=torch.long)
#         self.ptr = 0
#         self.full = False

#     def __len__(self) -> int:
#         return self.size if self.full else self.ptr

#     @torch.no_grad()
#     def add(self, feat: torch.Tensor, y_lang: torch.Tensor, y_len: torch.Tensor) -> None:
#         feat_cpu = feat.detach().float().cpu()
#         y_lang_cpu = y_lang.detach().long().cpu()
#         y_len_cpu = y_len.detach().long().cpu()

#         n = int(feat_cpu.size(0))
#         if n <= 0:
#             return

#         start = self.ptr
#         end = start + n

#         if end <= self.size:
#             self.feat[start:end].copy_(feat_cpu)
#             self.y_lang[start:end].copy_(y_lang_cpu)
#             self.y_len[start:end].copy_(y_len_cpu)
#             self.ptr = end
#             if self.ptr >= self.size:
#                 self.ptr = 0
#                 self.full = True
#         else:
#             first = self.size - start
#             self.feat[start:self.size].copy_(feat_cpu[:first])
#             self.y_lang[start:self.size].copy_(y_lang_cpu[:first])
#             self.y_len[start:self.size].copy_(y_len_cpu[:first])

#             remain = n - first
#             self.feat[0:remain].copy_(feat_cpu[first:])
#             self.y_lang[0:remain].copy_(y_lang_cpu[first:])
#             self.y_len[0:remain].copy_(y_len_cpu[first:])

#             self.ptr = remain
#             self.full = True

#     @torch.no_grad()
#     def sample(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         m = len(self)
#         if m <= 0:
#             raise RuntimeError("AdvFeatureBuffer is empty")

#         n_eff = min(int(n), m)
#         idx = torch.randint(low=0, high=m, size=(n_eff,), dtype=torch.long)
#         xb = self.feat[idx].to(device, non_blocking=True)
#         y_langb = self.y_lang[idx].to(device, non_blocking=True)
#         y_lenb = self.y_len[idx].to(device, non_blocking=True)
#         return xb, y_langb, y_lenb


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


# def save_checkpoint(
#     path: str,
#     model: nn.Module,
#     lang_clf: nn.Module,
#     len_clf: Optional[nn.Module],
#     opt_model: torch.optim.Optimizer,
#     opt_lang: torch.optim.Optimizer,
#     opt_len: Optional[torch.optim.Optimizer],
#     step: int,
#     cfg: Dict[str, Any],
# ) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     payload: Dict[str, Any] = {
#         "step": step,
#         "model_state": model.state_dict(),
#         "lang_clf_state": lang_clf.state_dict(),
#         "opt_model_state": opt_model.state_dict(),
#         "opt_lang_state": opt_lang.state_dict(),
#         "config": cfg,
#     }
#     if len_clf is not None:
#         payload["len_clf_state"] = len_clf.state_dict()
#     if opt_len is not None:
#         payload["opt_len_state"] = opt_len.state_dict()

#     torch.save(payload, path)


# def try_load_checkpoint(
#     resume_path: str,
#     model: nn.Module,
#     lang_clf: nn.Module,
#     len_clf: Optional[nn.Module],
#     opt_model: torch.optim.Optimizer,
#     opt_lang: torch.optim.Optimizer,
#     opt_len: Optional[torch.optim.Optimizer],
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

#     # Backward compatibility: old ckpt may not have len_clf/opt_len
#     if (len_clf is not None) and ("len_clf_state" in ckpt):
#         len_clf.load_state_dict(ckpt["len_clf_state"])
#     if (opt_len is not None) and ("opt_len_state" in ckpt):
#         opt_len.load_state_dict(ckpt["opt_len_state"])

#     step = int(ckpt.get("step", 0))
#     print(f"[RESUME] loaded {resume_path}, step={step}")
#     return step


# @torch.no_grad()
# def eval_lang_len_acc(
#     model: LatentRendererModel,
#     lang_clf: nn.Module,
#     len_clf: Optional[nn.Module],
#     dl: DataLoader,
#     device: torch.device,
#     max_batches: int = 200,
# ) -> Dict[str, float]:
#     model.eval()
#     lang_clf.eval()
#     if len_clf is not None:
#         len_clf.eval()

#     correct_lang = 0
#     total_lang = 0

#     correct_len = 0
#     total_len = 0

#     for i, batch in enumerate(dl):
#         en_ids = batch["en_input_ids"].to(device)
#         en_m = batch["en_attention_mask"].to(device)
#         zh_ids = batch["zh_input_ids"].to(device)
#         zh_m = batch["zh_attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         out = model(en_ids, en_m, zh_ids, zh_m, labels)
#         z_all = torch.cat([out.zbar_en, out.zbar_zh], dim=0)

#         y_lang = torch.cat(
#             [
#                 torch.zeros(out.zbar_en.size(0), dtype=torch.long, device=device),
#                 torch.ones(out.zbar_zh.size(0), dtype=torch.long, device=device),
#             ],
#             dim=0,
#         )
#         logits_lang = lang_clf(z_all)
#         pred_lang = logits_lang.argmax(dim=1)
#         correct_lang += int((pred_lang == y_lang).sum().item())
#         total_lang += int(y_lang.numel())

#         if len_clf is not None:
#             y_len = torch.cat([length_bucket_from_mask(en_m), length_bucket_from_mask(zh_m)], dim=0)
#             logits_len = len_clf(z_all)
#             pred_len = logits_len.argmax(dim=1)
#             correct_len += int((pred_len == y_len).sum().item())
#             total_len += int(y_len.numel())

#         if i + 1 >= max_batches:
#             break

#     model.train()
#     lang_clf.train()
#     if len_clf is not None:
#         len_clf.train()

#     out_dict = {"lang_acc_valid_evalmode": correct_lang / max(total_lang, 1)}
#     if len_clf is not None:
#         out_dict["len_acc_valid_evalmode"] = correct_len / max(total_len, 1)
#     return out_dict


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train_jsonl", default="data/groups_train.jsonl")
#     ap.add_argument("--valid_jsonl", default="data/groups_valid.jsonl")
#     ap.add_argument("--run_dir", default="runs/adv4_smoke")
#     ap.add_argument("--resume", default="", help="path to ckpt.pt")

#     ap.add_argument("--backbone", default="google/mt5-small")
#     ap.add_argument("--num_latents", type=int, default=16)

#     # keep these exposed; you currently set them to 0 in your local version
#     ap.add_argument("--latent_dropout", type=float, default=0.0)
#     ap.add_argument("--latent_noise_std", type=float, default=0.0)

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

#     # NEW: length adversary
#     ap.add_argument("--lambda_len", type=float, default=1.0, help="weight of length-bucket CE with GRL")

#     # optim
#     ap.add_argument("--lr_model", type=float, default=1e-4)
#     ap.add_argument("--lr_lang", type=float, default=1e-3)
#     ap.add_argument("--lr_len", type=float, default=1e-3)

#     # adversary training controls
#     ap.add_argument("--adv_clf_steps", type=int, default=2, help="clf updates per train step (from queue samples)")
#     ap.add_argument("--adv_clf_weight_decay", type=float, default=0.0)

#     ap.add_argument("--adv_queue_size", type=int, default=4096, help="ring buffer size (num samples)")
#     ap.add_argument("--adv_clf_batch", type=int, default=256, help="clf minibatch size sampled from queue")

#     args = ap.parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     os.makedirs(args.run_dir, exist_ok=True)

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

#     # language discriminator
#     lang_clf = nn.Sequential(
#         nn.Linear(d_model, d_model),
#         nn.ReLU(),
#         nn.Linear(d_model, 2),
#     ).to(device)

#     # length discriminator (4 buckets)
#     len_clf = nn.Sequential(
#         nn.Linear(d_model, d_model),
#         nn.ReLU(),
#         nn.Linear(d_model, 4),
#     ).to(device)

#     opt_model = torch.optim.AdamW(model.parameters(), lr=args.lr_model)
#     opt_lang = torch.optim.AdamW(lang_clf.parameters(), lr=args.lr_lang, weight_decay=args.adv_clf_weight_decay)
#     opt_len = torch.optim.AdamW(len_clf.parameters(), lr=args.lr_len, weight_decay=args.adv_clf_weight_decay)

#     # resume (supports old ckpt that do not contain len_clf/opt_len)
#     step = try_load_checkpoint(args.resume, model, lang_clf, len_clf, opt_model, opt_lang, opt_len, device)

#     # feature buffer
#     buf = AdvFeatureBuffer(size=args.adv_queue_size, dim=d_model)

#     t0 = time.time()
#     log_path = os.path.join(args.run_dir, "logs.jsonl")
#     ckpt_path = os.path.join(args.run_dir, "ckpt.pt")

#     random.seed(42)
#     torch.manual_seed(42)

#     opt_model.zero_grad(set_to_none=True)
#     opt_lang.zero_grad(set_to_none=True)
#     opt_len.zero_grad(set_to_none=True)

#     for epoch in range(args.epochs):
#         model.train()
#         lang_clf.train()
#         len_clf.train()

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
#             y_len = torch.cat([length_bucket_from_mask(en_m), length_bucket_from_mask(zh_m)], dim=0)

#             # push to buffer
#             buf.add(z_all, y_lang, y_len)

#             # render losses
#             loss_en = ce_loss_from_logits(out.logits_en, labels)
#             loss_zh = ce_loss_from_logits(out.logits_zh, labels)
#             loss_render = 0.5 * (loss_en + loss_zh)

#             # align + regularizers
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

#             # adversaries
#             loss_lang = torch.tensor(0.0, device=device)
#             loss_len = torch.tensor(0.0, device=device)

#             acc_lang_batch = 0.0
#             acc_lang_clf_detached = 0.0
#             acc_len_batch = 0.0
#             acc_len_clf_detached = 0.0

#             alpha = 0.0

#             adv_active = (step >= args.adv_start_step) and ((args.lambda_lang > 0) or (args.lambda_len > 0)) and (len(buf) > 0)

#             if adv_active:
#                 # 1) train discriminators on queue samples (detached by construction)
#                 set_requires_grad(lang_clf, True)
#                 set_requires_grad(len_clf, True)

#                 for _ in range(max(args.adv_clf_steps, 1)):
#                     xb, y_langb, y_lenb = buf.sample(args.adv_clf_batch, device=device)

#                     if args.lambda_lang > 0:
#                         opt_lang.zero_grad(set_to_none=True)
#                         logits_lang_det = lang_clf(xb)
#                         loss_clf_lang = F.cross_entropy(logits_lang_det, y_langb)
#                         loss_clf_lang.backward()
#                         opt_lang.step()
#                         with torch.no_grad():
#                             acc_lang_clf_detached = (logits_lang_det.argmax(dim=1) == y_langb).float().mean().item()

#                     if args.lambda_len > 0:
#                         opt_len.zero_grad(set_to_none=True)
#                         logits_len_det = len_clf(xb)
#                         loss_clf_len = F.cross_entropy(logits_len_det, y_lenb)
#                         loss_clf_len.backward()
#                         opt_len.step()
#                         with torch.no_grad():
#                             acc_len_clf_detached = (logits_len_det.argmax(dim=1) == y_lenb).float().mean().item()

#                 # 2) encoder adversarial step using current batch
#                 k = min(max(step - args.adv_start_step, 0), args.grl_warmup)
#                 alpha = args.grl_alpha * (k / max(args.grl_warmup, 1))

#                 # freeze clf params so gradients go only into encoder through z_all
#                 set_requires_grad(lang_clf, False)
#                 set_requires_grad(len_clf, False)

#                 z_rev = grad_reverse(z_all, alpha)

#                 if args.lambda_lang > 0:
#                     logits_adv_lang = lang_clf(z_rev)
#                     loss_lang = F.cross_entropy(logits_adv_lang, y_lang)
#                     with torch.no_grad():
#                         acc_lang_batch = (logits_adv_lang.argmax(dim=1) == y_lang).float().mean().item()

#                 if args.lambda_len > 0:
#                     logits_adv_len = len_clf(z_rev)
#                     loss_len = F.cross_entropy(logits_adv_len, y_len)
#                     with torch.no_grad():
#                         acc_len_batch = (logits_adv_len.argmax(dim=1) == y_len).float().mean().item()

#                 set_requires_grad(lang_clf, True)
#                 set_requires_grad(len_clf, True)

#             loss = (
#                 loss_render
#                 + args.lambda_align * loss_align
#                 + args.lambda_varcov * loss_varcov
#                 + args.lambda_mean * loss_mean
#                 + args.lambda_mean_diff * loss_mean_diff
#                 + args.lambda_pair * loss_pair
#                 + args.lambda_lang * loss_lang
#                 + args.lambda_len * loss_len
#             )

#             (loss / args.grad_accum).backward()

#             if step % args.grad_accum == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 opt_model.step()
#                 opt_model.zero_grad(set_to_none=True)

#             if step % 20 == 0:
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
#                     "loss_len": float(loss_len.item()),
#                     "grl_alpha_eff": float(alpha),
#                     "lang_acc_batch": float(acc_lang_batch),
#                     "lang_acc_clf_detached": float(acc_lang_clf_detached),
#                     "len_acc_batch": float(acc_len_batch),
#                     "len_acc_clf_detached": float(acc_len_clf_detached),
#                     "adv_buf_fill": int(len(buf)),
#                     "elapsed_sec": float(time.time() - t0),
#                 }
#                 print(msg)
#                 with open(log_path, "a", encoding="utf-8") as f:
#                     f.write(json.dumps(msg) + "\n")

#             if step % args.eval_every == 0:
#                 diag = {"step": step, **eval_nll(model, valid_dl, device)}
#                 diag.update(eval_lang_len_acc(model, lang_clf, len_clf, valid_dl, device))
#                 print("EVAL:", diag)
#                 with open(os.path.join(args.run_dir, "diag.jsonl"), "a", encoding="utf-8") as f:
#                     f.write(json.dumps(diag) + "\n")
#                 save_checkpoint(ckpt_path, model, lang_clf, len_clf, opt_model, opt_lang, opt_len, step, cfg)

#     save_checkpoint(ckpt_path, model, lang_clf, len_clf, opt_model, opt_lang, opt_len, step, cfg)
#     print(f"Done. Saved checkpoint to {ckpt_path}")


# if __name__ == "__main__":
#     main()


# src/train_planB_ramp_adv.py
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Flags:
    TRAIN_ENTRY: str = os.environ.get("TRAIN_ENTRY", "src/train_adv4.py")

    RUN_DIR_FLAG: str = "--run_dir"
    RESUME_FLAG: str = "--resume"

    BATCH_SIZE_FLAG: str = "--batch_size"
    EPOCHS_FLAG: str = "--epochs"
    LR_FLAG: str = "--lr_model"

    L_ALIGN_FLAG: str = "--lambda_align"
    L_VARCOV_FLAG: str = "--lambda_varcov"
    L_MEAN_FLAG: str = "--lambda_mean"
    L_MEAN_DIFF_FLAG: str = "--lambda_mean_diff"
    L_PAIR_FLAG: str = "--lambda_pair"

    L_LANG_FLAG: str = "--lambda_lang"
    L_LEN_FLAG: str = "--lambda_len"

    ADV_START_FLAG: str = "--adv_start_step"
    GRL_WARMUP_FLAG: str = "--grl_warmup"
    GRL_ALPHA_FLAG: str = "--grl_alpha"

    LANG_CLF_LR_FLAG: str = "--lr_lang"
    LEN_CLF_LR_FLAG: str = "--lr_len"

    ADV_CLF_STEPS_FLAG: str = "--adv_clf_steps"
    ADV_QUEUE_SIZE_FLAG: str = "--adv_queue_size"
    ADV_CLF_BATCH_FLAG: str = "--adv_clf_batch"

    # NEW
    ADV_MIX_CURRENT_FLAG: str = "--adv_mix_current"


def run(cmd: list[str]) -> None:
    print("\n[CMD]")
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def stage_cmd(
    flags: Flags,
    *,
    run_dir: str,
    resume: str,
    batch_size: int,
    epochs: int,
    lr: float,
    lambda_align: float,
    lambda_varcov: float,
    lambda_mean: float,
    lambda_mean_diff: float,
    lambda_pair: float,
    lambda_lang: float,
    lambda_len: float,
    adv_start_step: int,
    grl_warmup: int,
    grl_alpha: float,
    lang_clf_lr: float,
    len_clf_lr: float,
    adv_clf_steps: int,
    adv_queue_size: int,
    adv_clf_batch: int,
    adv_mix_current: float,
) -> list[str]:
    cmd = [
        sys.executable,
        flags.TRAIN_ENTRY,
        flags.RUN_DIR_FLAG, run_dir,
        flags.BATCH_SIZE_FLAG, str(batch_size),
        flags.EPOCHS_FLAG, str(epochs),
        flags.LR_FLAG, str(lr),

        flags.L_ALIGN_FLAG, str(lambda_align),
        flags.L_VARCOV_FLAG, str(lambda_varcov),
        flags.L_MEAN_FLAG, str(lambda_mean),
        flags.L_MEAN_DIFF_FLAG, str(lambda_mean_diff),
        flags.L_PAIR_FLAG, str(lambda_pair),

        flags.L_LANG_FLAG, str(lambda_lang),
        flags.L_LEN_FLAG, str(lambda_len),

        flags.ADV_START_FLAG, str(adv_start_step),
        flags.GRL_WARMUP_FLAG, str(grl_warmup),
        flags.GRL_ALPHA_FLAG, str(grl_alpha),

        flags.LANG_CLF_LR_FLAG, str(lang_clf_lr),
        flags.LEN_CLF_LR_FLAG, str(len_clf_lr),

        flags.ADV_CLF_STEPS_FLAG, str(adv_clf_steps),
        flags.ADV_QUEUE_SIZE_FLAG, str(adv_queue_size),
        flags.ADV_CLF_BATCH_FLAG, str(adv_clf_batch),

        flags.ADV_MIX_CURRENT_FLAG, str(adv_mix_current),
    ]
    if resume:
        cmd += [flags.RESUME_FLAG, resume]
    return cmd


def main():
    f = Flags()

    run_dir = os.environ.get("RUN_DIR", "runs/planB_ramp_adv")
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    lr = float(os.environ.get("LR", "3e-4"))

    lambda_align = float(os.environ.get("LAMBDA_ALIGN", "1.0"))
    lambda_varcov = float(os.environ.get("LAMBDA_VARCOV", "0.1"))
    lambda_mean = float(os.environ.get("LAMBDA_MEAN", "0.5"))
    lambda_mean_diff = float(os.environ.get("LAMBDA_MEAN_DIFF", "0.1"))
    lambda_pair = float(os.environ.get("LAMBDA_PAIR", "0.005"))

    lambda_lang_warm = float(os.environ.get("LAMBDA_LANG_WARMUP", "1.0"))
    lambda_lang_ramp = float(os.environ.get("LAMBDA_LANG_RAMP", "1.0"))
    lambda_lang_final = float(os.environ.get("LAMBDA_LANG_FINAL", "2.0"))

    lambda_len_warm = float(os.environ.get("LAMBDA_LEN_WARMUP", "1.0"))
    lambda_len_ramp = float(os.environ.get("LAMBDA_LEN_RAMP", "1.0"))
    lambda_len_final = float(os.environ.get("LAMBDA_LEN_FINAL", "2.0"))

    lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "0.001"))
    len_clf_lr = float(os.environ.get("LEN_CLF_LR", "0.001"))

    adv_queue_size = int(os.environ.get("ADV_QUEUE_SIZE", "4096"))
    adv_clf_batch = int(os.environ.get("ADV_CLF_BATCH", "256"))

    # NEW
    adv_mix_current = float(os.environ.get("ADV_MIX_CURRENT", "0.5"))

    ckpt_path = f"{run_dir}/ckpt.pt"

    warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", "1"))
    ramp_epochs = int(os.environ.get("RAMP_EPOCHS", "2"))
    final_epochs = int(os.environ.get("FINAL_EPOCHS", "5"))

    grl_warmup = int(os.environ.get("GRL_WARMUP", "200"))

    cmd1 = stage_cmd(
        f,
        run_dir=run_dir,
        resume="",
        batch_size=batch_size,
        epochs=warmup_epochs,
        lr=lr,
        lambda_align=lambda_align,
        lambda_varcov=lambda_varcov,
        lambda_mean=0.0,
        lambda_mean_diff=0.0,
        lambda_pair=0.0,
        lambda_lang=lambda_lang_warm,
        lambda_len=lambda_len_warm,
        adv_start_step=0,
        grl_warmup=grl_warmup,
        grl_alpha=0.0,
        lang_clf_lr=lang_clf_lr,
        len_clf_lr=len_clf_lr,
        adv_clf_steps=int(os.environ.get("ADV_STEPS_WARMUP", "4")),
        adv_queue_size=adv_queue_size,
        adv_clf_batch=adv_clf_batch,
        adv_mix_current=adv_mix_current,
    )
    run(cmd1)

    cmd2 = stage_cmd(
        f,
        run_dir=run_dir,
        resume=ckpt_path,
        batch_size=batch_size,
        epochs=ramp_epochs,
        lr=lr,
        lambda_align=lambda_align,
        lambda_varcov=lambda_varcov,
        lambda_mean=lambda_mean,
        lambda_mean_diff=lambda_mean_diff,
        lambda_pair=lambda_pair,
        lambda_lang=lambda_lang_ramp,
        lambda_len=lambda_len_ramp,
        adv_start_step=0,
        grl_warmup=grl_warmup,
        grl_alpha=float(os.environ.get("GRL_ALPHA_RAMP", "1.0")),
        lang_clf_lr=lang_clf_lr,
        len_clf_lr=len_clf_lr,
        adv_clf_steps=int(os.environ.get("ADV_STEPS_RAMP", "3")),
        adv_queue_size=adv_queue_size,
        adv_clf_batch=adv_clf_batch,
        adv_mix_current=adv_mix_current,
    )
    run(cmd2)

    cmd3 = stage_cmd(
        f,
        run_dir=run_dir,
        resume=ckpt_path,
        batch_size=batch_size,
        epochs=final_epochs,
        lr=lr,
        lambda_align=lambda_align,
        lambda_varcov=lambda_varcov,
        lambda_mean=lambda_mean,
        lambda_mean_diff=lambda_mean_diff,
        lambda_pair=lambda_pair,
        lambda_lang=lambda_lang_final,
        lambda_len=lambda_len_final,
        adv_start_step=0,
        grl_warmup=grl_warmup,
        grl_alpha=float(os.environ.get("GRL_ALPHA_FINAL", "2.0")),
        lang_clf_lr=lang_clf_lr,
        len_clf_lr=len_clf_lr,
        adv_clf_steps=int(os.environ.get("ADV_STEPS_FINAL", "2")),
        adv_queue_size=adv_queue_size,
        adv_clf_batch=adv_clf_batch,
        adv_mix_current=adv_mix_current,
    )
    run(cmd3)

    print("\n[Next]")
    print(f"python src/eval_diag.py --ckpt {ckpt_path} --batch_size 8")
    print(f"python src/probe_leakage.py --ckpt {ckpt_path} --max_train_groups 8000 --max_valid_groups 946")


if __name__ == "__main__":
    main()
