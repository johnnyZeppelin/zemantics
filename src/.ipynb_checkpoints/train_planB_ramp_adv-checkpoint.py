# src/train_planB_ramp_adv.py
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Flags:
    TRAIN_ENTRY: str = os.environ.get("TRAIN_ENTRY", "src/train_adv5.py")

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

    # lambda_len_warm = float(os.environ.get("LAMBDA_LEN_WARMUP", "1.0"))
    # lambda_len_ramp = float(os.environ.get("LAMBDA_LEN_RAMP", "1.0"))
    # lambda_len_final = float(os.environ.get("LAMBDA_LEN_FINAL", "2.0"))
    lambda_len_warm = float(os.environ.get("LAMBDA_LEN_WARMUP", "0.0"))
    lambda_len_ramp = float(os.environ.get("LAMBDA_LEN_RAMP", "0.0"))
    lambda_len_final = float(os.environ.get("LAMBDA_LEN_FINAL", "0.0"))

    # lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "0.001"))
    lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "0.005"))
    len_clf_lr = float(os.environ.get("LEN_CLF_LR", "0.001"))

    adv_queue_size = int(os.environ.get("ADV_QUEUE_SIZE", "4096"))
    # adv_clf_batch = int(os.environ.get("ADV_CLF_BATCH", "256"))
    adv_clf_batch = int(os.environ.get("ADV_CLF_BATCH", "1024"))

    # NEW
    # adv_mix_current = float(os.environ.get("ADV_MIX_CURRENT", "0.5"))
    adv_mix_current = float(os.environ.get("ADV_MIX_CURRENT", "0.2"))

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
        # adv_clf_steps=int(os.environ.get("ADV_STEPS_WARMUP", "4")),
        adv_clf_steps=int(os.environ.get("ADV_STEPS_WARMUP", "10")),
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
        # adv_clf_steps=int(os.environ.get("ADV_STEPS_RAMP", "3")),
        adv_clf_steps=int(os.environ.get("ADV_STEPS_RAMP", "10")),
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
        # adv_clf_steps=int(os.environ.get("ADV_STEPS_FINAL", "2")),
        adv_clf_steps=int(os.environ.get("ADV_STEPS_FINAL", "20")),
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
