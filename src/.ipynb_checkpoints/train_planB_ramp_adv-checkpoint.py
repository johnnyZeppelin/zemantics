# # src/train_planB_ramp_adv.py
# import os
# import shlex
# import subprocess
# import sys
# from dataclasses import dataclass


# @dataclass
# class Flags:
#     TRAIN_ENTRY: str = os.environ.get("TRAIN_ENTRY", "src/train_adv.py")

#     RUN_DIR_FLAG: str = os.environ.get("RUN_DIR_FLAG", "--run_dir")
#     RESUME_FLAG: str = os.environ.get("RESUME_FLAG", "--resume")  # 改成你实际用的，如 --resume_ckpt
#     SEED_FLAG: str = os.environ.get("SEED_FLAG", "--seed")
#     BATCH_SIZE_FLAG: str = os.environ.get("BATCH_SIZE_FLAG", "--batch_size")
#     MAX_STEPS_FLAG: str = os.environ.get("MAX_STEPS_FLAG", "--max_steps")
#     LR_FLAG: str = os.environ.get("LR_FLAG", "--lr")
#     WD_FLAG: str = os.environ.get("WD_FLAG", "--weight_decay")
#     GRAD_CLIP_FLAG: str = os.environ.get("GRAD_CLIP_FLAG", "--grad_clip")

#     L_ALIGN_FLAG: str = os.environ.get("L_ALIGN_FLAG", "--lambda_align")
#     L_VARCOV_FLAG: str = os.environ.get("L_VARCOV_FLAG", "--lambda_varcov")
#     L_MEAN_FLAG: str = os.environ.get("L_MEAN_FLAG", "--lambda_mean")
#     L_MEAN_DIFF_FLAG: str = os.environ.get("L_MEAN_DIFF_FLAG", "--lambda_mean_diff")
#     L_PAIR_FLAG: str = os.environ.get("L_PAIR_FLAG", "--lambda_pair")
#     L_LANG_FLAG: str = os.environ.get("L_LANG_FLAG", "--lambda_lang")

#     GRL_ALPHA_FLAG: str = os.environ.get("GRL_ALPHA_FLAG", "--grl_alpha")
#     LANG_CLF_LR_FLAG: str = os.environ.get("LANG_CLF_LR_FLAG", "--lang_clf_lr")
#     LANG_CLF_STEPS_FLAG: str = os.environ.get("LANG_CLF_STEPS_FLAG", "--lang_clf_steps")


# def run(cmd: list[str]) -> None:
#     print("\n[CMD]")
#     print(" ".join(shlex.quote(x) for x in cmd))
#     subprocess.run(cmd, check=True)


# def stage_cmd(flags: Flags, *, run_dir: str, seed: int, batch_size: int, max_steps: int,
#               lr: float, wd: float, grad_clip: float,
#               lambda_align: float, lambda_varcov: float, lambda_mean: float, lambda_mean_diff: float, lambda_pair: float,
#               lambda_lang: float, grl_alpha: float, lang_clf_lr: float, lang_clf_steps: int,
#               resume_ckpt: str | None) -> list[str]:
#     cmd = [
#         sys.executable, flags.TRAIN_ENTRY,
#         flags.RUN_DIR_FLAG, run_dir,
#         flags.SEED_FLAG, str(seed),
#         flags.BATCH_SIZE_FLAG, str(batch_size),
#         flags.MAX_STEPS_FLAG, str(max_steps),
#         flags.LR_FLAG, str(lr),
#         flags.WD_FLAG, str(wd),
#         flags.GRAD_CLIP_FLAG, str(grad_clip),

#         flags.L_ALIGN_FLAG, str(lambda_align),
#         flags.L_VARCOV_FLAG, str(lambda_varcov),
#         flags.L_MEAN_FLAG, str(lambda_mean),
#         flags.L_MEAN_DIFF_FLAG, str(lambda_mean_diff),
#         flags.L_PAIR_FLAG, str(lambda_pair),

#         flags.L_LANG_FLAG, str(lambda_lang),
#         flags.GRL_ALPHA_FLAG, str(grl_alpha),
#         flags.LANG_CLF_LR_FLAG, str(lang_clf_lr),
#         flags.LANG_CLF_STEPS_FLAG, str(lang_clf_steps),
#     ]
#     if resume_ckpt:
#         cmd += [flags.RESUME_FLAG, resume_ckpt]
#     return cmd


# def main():
#     f = Flags()

#     run_dir = os.environ.get("RUN_DIR", "runs/planB_ramp_adv")
#     seed = int(os.environ.get("SEED", "0"))
#     batch_size = int(os.environ.get("BATCH_SIZE", "8"))

#     # 三阶段步数（都是 “max_steps=累计步数” 的风格）
#     warmup_steps = int(os.environ.get("WARMUP_STEPS", "2000"))
#     ramp_steps = int(os.environ.get("RAMP_STEPS", "4000"))
#     final_steps = int(os.environ.get("FINAL_STEPS", "20000"))

#     lr = float(os.environ.get("LR", "3e-4"))
#     wd = float(os.environ.get("WD", "0.01"))
#     grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))

#     # 共同正则项，B 方案倾向温和一些，先保 inv / nll
#     lambda_align = float(os.environ.get("LAMBDA_ALIGN", "1.0"))
#     lambda_varcov = float(os.environ.get("LAMBDA_VARCOV", "0.1"))
#     lambda_mean = float(os.environ.get("LAMBDA_MEAN", "0.5"))
#     lambda_mean_diff = float(os.environ.get("LAMBDA_MEAN_DIFF", "0.1"))
#     lambda_pair = float(os.environ.get("LAMBDA_PAIR", "0.005"))

#     lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "5e-4"))
#     lang_clf_steps = int(os.environ.get("LANG_CLF_STEPS", "1"))

#     ckpt_path = f"{run_dir}/ckpt.pt"

#     # Stage 1: warmup, 不做语言对抗，让基本任务先收敛一点
#     cmd1 = stage_cmd(
#         f, run_dir=run_dir, seed=seed, batch_size=batch_size, max_steps=warmup_steps,
#         lr=lr, wd=wd, grad_clip=grad_clip,
#         lambda_align=lambda_align, lambda_varcov=lambda_varcov,
#         lambda_mean=0.0, lambda_mean_diff=0.0, lambda_pair=0.0,
#         lambda_lang=0.0, grl_alpha=0.0,
#         lang_clf_lr=lang_clf_lr, lang_clf_steps=lang_clf_steps,
#         resume_ckpt=None
#     )
#     run(cmd1)

#     # Stage 2: ramp, 逐渐加对抗但不拉满
#     # 这里用中等强度，目标是把 probe 先从 ~0.996 拉到更低，同时 inv 不要崩
#     cmd2 = stage_cmd(
#         f, run_dir=run_dir, seed=seed, batch_size=batch_size, max_steps=ramp_steps,
#         lr=lr, wd=wd, grad_clip=grad_clip,
#         lambda_align=lambda_align, lambda_varcov=lambda_varcov,
#         lambda_mean=lambda_mean, lambda_mean_diff=lambda_mean_diff, lambda_pair=lambda_pair,
#         lambda_lang=float(os.environ.get("LAMBDA_LANG_RAMP", "1.0")),
#         grl_alpha=float(os.environ.get("GRL_ALPHA_RAMP", "1.0")),
#         lang_clf_lr=lang_clf_lr, lang_clf_steps=lang_clf_steps,
#         resume_ckpt=ckpt_path
#     )
#     run(cmd2)

#     # Stage 3: final, 稳态对抗，略增强
#     cmd3 = stage_cmd(
#         f, run_dir=run_dir, seed=seed, batch_size=batch_size, max_steps=final_steps,
#         lr=lr, wd=wd, grad_clip=grad_clip,
#         lambda_align=lambda_align, lambda_varcov=lambda_varcov,
#         lambda_mean=lambda_mean, lambda_mean_diff=lambda_mean_diff, lambda_pair=lambda_pair,
#         lambda_lang=float(os.environ.get("LAMBDA_LANG_FINAL", "2.0")),
#         grl_alpha=float(os.environ.get("GRL_ALPHA_FINAL", "2.0")),
#         lang_clf_lr=lang_clf_lr, lang_clf_steps=lang_clf_steps,
#         resume_ckpt=ckpt_path
#     )
#     run(cmd3)

#     print("\n[Next]")
#     print(f"python src/eval_diag.py --ckpt {ckpt_path} --batch_size 8")
#     print(f"python src/probe_leakage.py --ckpt {ckpt_path} --max_train_groups 8000 --max_valid_groups 946")


# if __name__ == "__main__":
#     main()

# # src/train_planB_ramp_adv.py
# import os
# import shlex
# import subprocess
# import sys
# from dataclasses import dataclass

# @dataclass
# class Flags:
#     TRAIN_ENTRY: str = os.environ.get("TRAIN_ENTRY", "src/train_adv4.py")

#     RUN_DIR_FLAG: str = os.environ.get("RUN_DIR_FLAG", "--run_dir")
#     # Note: train_adv4.py currently doesn't support --resume. 
#     # Commenting out to avoid "unrecognized argument" error.
#     # RESUME_FLAG: str = os.environ.get("RESUME_FLAG", "--resume") 
    
#     BATCH_SIZE_FLAG: str = os.environ.get("BATCH_SIZE_FLAG", "--batch_size")
#     EPOCHS_FLAG: str = os.environ.get("EPOCHS_FLAG", "--epochs") # train_adv4 uses epochs
    
#     LR_FLAG: str = os.environ.get("LR_FLAG", "--lr_model") # Fixed ambiguity
    
#     L_ALIGN_FLAG: str = os.environ.get("L_ALIGN_FLAG", "--lambda_align")
#     L_VARCOV_FLAG: str = os.environ.get("L_VARCOV_FLAG", "--lambda_varcov")
#     L_MEAN_FLAG: str = os.environ.get("L_MEAN_FLAG", "--lambda_mean")
#     L_MEAN_DIFF_FLAG: str = os.environ.get("L_MEAN_DIFF_FLAG", "--lambda_mean_diff")
#     L_PAIR_FLAG: str = os.environ.get("L_PAIR_FLAG", "--lambda_pair")
#     L_LANG_FLAG: str = os.environ.get("L_LANG_FLAG", "--lambda_lang")

#     GRL_ALPHA_FLAG: str = os.environ.get("GRL_ALPHA_FLAG", "--grl_alpha")
#     LANG_CLF_LR_FLAG: str = os.environ.get("LANG_CLF_LR_FLAG", "--lr_lang") # Fixed name

# def run(cmd: list[str]) -> None:
#     print("\n[CMD]")
#     print(" ".join(shlex.quote(x) for x in cmd))
#     subprocess.run(cmd, check=True)

# def stage_cmd(flags: Flags, *, run_dir: str, batch_size: int, epochs: int,
#               lr: float, lambda_align: float, lambda_varcov: float, 
#               lambda_mean: float, lambda_mean_diff: float, lambda_pair: float,
#               lambda_lang: float, grl_alpha: float, lang_clf_lr: float) -> list[str]:
#     cmd = [
#         sys.executable, flags.TRAIN_ENTRY,
#         flags.RUN_DIR_FLAG, run_dir,
#         flags.BATCH_SIZE_FLAG, str(batch_size),
#         flags.EPOCHS_FLAG, str(epochs),
#         flags.LR_FLAG, str(lr),
#         flags.L_ALIGN_FLAG, str(lambda_align),
#         flags.L_VARCOV_FLAG, str(lambda_varcov),
#         flags.L_MEAN_FLAG, str(lambda_mean),
#         flags.L_MEAN_DIFF_FLAG, str(lambda_mean_diff),
#         flags.L_PAIR_FLAG, str(lambda_pair),
#         flags.L_LANG_FLAG, str(lambda_lang),
#         flags.GRL_ALPHA_FLAG, str(grl_alpha),
#         flags.LANG_CLF_LR_FLAG, str(lang_clf_lr),
#     ]
#     return cmd

# def main():
#     f = Flags()

#     run_dir = os.environ.get("RUN_DIR", "runs/planB_ramp_adv")
#     batch_size = int(os.environ.get("BATCH_SIZE", "8"))

#     # train_adv4 uses epochs. Setting small epoch counts for stages 
#     # as proxy for steps if you aren't modifying train_adv4.py
#     warmup_epochs = 1 
#     ramp_epochs = 2
#     final_epochs = 5

#     lr = float(os.environ.get("LR", "3e-4"))
#     lambda_align = float(os.environ.get("LAMBDA_ALIGN", "1.0"))
#     lambda_varcov = float(os.environ.get("LAMBDA_VARCOV", "0.1"))
#     lambda_mean = float(os.environ.get("LAMBDA_MEAN", "0.5"))
#     lambda_mean_diff = float(os.environ.get("LAMBDA_MEAN_DIFF", "0.1"))
#     lambda_pair = float(os.environ.get("LAMBDA_PAIR", "0.005"))
#     lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "5e-4"))

#     ckpt_path = f"{run_dir}/ckpt.pt"

#     # Stage 1: warmup
#     cmd1 = stage_cmd(
#         f, run_dir=run_dir, batch_size=batch_size, epochs=warmup_epochs,
#         lr=lr, lambda_align=lambda_align, lambda_varcov=lambda_varcov,
#         lambda_mean=0.0, lambda_mean_diff=0.0, lambda_pair=0.0,
#         lambda_lang=0.0, grl_alpha=0.0, lang_clf_lr=lang_clf_lr
#     )
#     run(cmd1)

#     # Stage 2: ramp
#     cmd2 = stage_cmd(
#         f, run_dir=run_dir, batch_size=batch_size, epochs=ramp_epochs,
#         lr=lr, lambda_align=lambda_align, lambda_varcov=lambda_varcov,
#         lambda_mean=lambda_mean, lambda_mean_diff=lambda_mean_diff, lambda_pair=lambda_pair,
#         lambda_lang=float(os.environ.get("LAMBDA_LANG_RAMP", "1.0")),
#         grl_alpha=float(os.environ.get("GRL_ALPHA_RAMP", "1.0")),
#         lang_clf_lr=lang_clf_lr
#     )
#     run(cmd2)

#     # Stage 3: final
#     cmd3 = stage_cmd(
#         f, run_dir=run_dir, batch_size=batch_size, epochs=final_epochs,
#         lr=lr, lambda_align=lambda_align, lambda_varcov=lambda_varcov,
#         lambda_mean=lambda_mean, lambda_mean_diff=lambda_mean_diff, lambda_pair=lambda_pair,
#         lambda_lang=float(os.environ.get("LAMBDA_LANG_FINAL", "2.0")),
#         grl_alpha=float(os.environ.get("GRL_ALPHA_FINAL", "2.0")),
#         lang_clf_lr=lang_clf_lr
#     )
#     run(cmd3)

#     print("\n[Next]")
#     print(f"python src/eval_diag.py --ckpt {ckpt_path} --batch_size 8")
#     print(f"python src/probe_leakage.py --ckpt {run_dir}/ckpt.pt --max_train_groups 8000 --max_valid_groups 946")

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

    ADV_START_FLAG: str = "--adv_start_step"
    GRL_WARMUP_FLAG: str = "--grl_warmup"
    GRL_ALPHA_FLAG: str = "--grl_alpha"
    LANG_CLF_LR_FLAG: str = "--lr_lang"


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
    adv_start_step: int,
    grl_warmup: int,
    grl_alpha: float,
    lang_clf_lr: float,
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
        flags.ADV_START_FLAG, str(adv_start_step),
        flags.GRL_WARMUP_FLAG, str(grl_warmup),
        flags.GRL_ALPHA_FLAG, str(grl_alpha),
        flags.LANG_CLF_LR_FLAG, str(lang_clf_lr),
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
    # lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "5e-4"))
    lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "0.001"))

    ckpt_path = f"{run_dir}/ckpt.pt"

    # 你原来 1/2/5 个 epoch，这里保留
    warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", "1"))
    ramp_epochs = int(os.environ.get("RAMP_EPOCHS", "2"))
    final_epochs = int(os.environ.get("FINAL_EPOCHS", "5"))

    # Stage 1: 预热 lang_clf（alpha=0，不反传 encoder），立即开始训练 clf
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
        lambda_lang=float(os.environ.get("LAMBDA_LANG_WARMUP", "1.0")),
        adv_start_step=0,
        grl_warmup=int(os.environ.get("GRL_WARMUP", "200")),
        grl_alpha=0.0,
        lang_clf_lr=lang_clf_lr,
    )
    run(cmd1)

    # Stage 2: ramp（resume），开始把 alpha 拉起来
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
        lambda_lang=float(os.environ.get("LAMBDA_LANG_RAMP", "1.0")),
        adv_start_step=0,
        grl_warmup=int(os.environ.get("GRL_WARMUP", "200")),
        grl_alpha=float(os.environ.get("GRL_ALPHA_RAMP", "1.0")),
        lang_clf_lr=lang_clf_lr,
    )
    run(cmd2)

    # Stage 3: final（resume）
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
        lambda_lang=float(os.environ.get("LAMBDA_LANG_FINAL", "2.0")),
        adv_start_step=0,
        grl_warmup=int(os.environ.get("GRL_WARMUP", "200")),
        grl_alpha=float(os.environ.get("GRL_ALPHA_FINAL", "2.0")),
        lang_clf_lr=lang_clf_lr,
    )
    run(cmd3)

    print("\n[Next]")
    print(f"python src/eval_diag.py --ckpt {ckpt_path} --batch_size 8")
    print(f"python src/probe_leakage.py --ckpt {ckpt_path} --max_train_groups 8000 --max_valid_groups 946")


if __name__ == "__main__":
    main()
