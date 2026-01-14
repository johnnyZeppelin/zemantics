# # # src/train_planA_strong_adv.py
# # import os
# # import shlex
# # import subprocess
# # import sys
# # from dataclasses import dataclass


# # @dataclass
# # class Flags:
# #     # 你的训练入口脚本路径，比如 "src/train_adv.py" 或 "src/train_bottleneck_adv.py"
# #     TRAIN_ENTRY: str = os.environ.get("TRAIN_ENTRY", "src/train_adv.py")

# #     # 下面这些 flag 名可以按需用环境变量覆盖，避免你每次改代码
# #     RUN_DIR_FLAG: str = os.environ.get("RUN_DIR_FLAG", "--run_dir")
# #     SEED_FLAG: str = os.environ.get("SEED_FLAG", "--seed")
# #     BATCH_SIZE_FLAG: str = os.environ.get("BATCH_SIZE_FLAG", "--batch_size")
# #     MAX_STEPS_FLAG: str = os.environ.get("MAX_STEPS_FLAG", "--max_steps")
# #     LR_FLAG: str = os.environ.get("LR_FLAG", "--lr")
# #     WD_FLAG: str = os.environ.get("WD_FLAG", "--weight_decay")
# #     GRAD_CLIP_FLAG: str = os.environ.get("GRAD_CLIP_FLAG", "--grad_clip")

# #     # loss 权重相关 flag（按你现有实现改名即可）
# #     L_ALIGN_FLAG: str = os.environ.get("L_ALIGN_FLAG", "--lambda_align")
# #     L_VARCOV_FLAG: str = os.environ.get("L_VARCOV_FLAG", "--lambda_varcov")
# #     L_MEAN_FLAG: str = os.environ.get("L_MEAN_FLAG", "--lambda_mean")
# #     L_MEAN_DIFF_FLAG: str = os.environ.get("L_MEAN_DIFF_FLAG", "--lambda_mean_diff")
# #     L_PAIR_FLAG: str = os.environ.get("L_PAIR_FLAG", "--lambda_pair")
# #     L_LANG_FLAG: str = os.environ.get("L_LANG_FLAG", "--lambda_lang")

# #     # 语言对抗机制相关 flag（如果你用 GRL 或者 encoder 反向梯度系数）
# #     GRL_ALPHA_FLAG: str = os.environ.get("GRL_ALPHA_FLAG", "--grl_alpha")
# #     LANG_CLF_LR_FLAG: str = os.environ.get("LANG_CLF_LR_FLAG", "--lang_clf_lr")
# #     LANG_CLF_STEPS_FLAG: str = os.environ.get("LANG_CLF_STEPS_FLAG", "--lang_clf_steps")


# # def run(cmd: list[str]) -> None:
# #     print("\n[CMD]")
# #     print(" ".join(shlex.quote(x) for x in cmd))
# #     subprocess.run(cmd, check=True)


# # def main():
# #     f = Flags()

# #     # 你也可以用环境变量覆盖这些默认超参
# #     run_dir = os.environ.get("RUN_DIR", "runs/planA_strong_adv")
# #     seed = int(os.environ.get("SEED", "0"))
# #     batch_size = int(os.environ.get("BATCH_SIZE", "8"))
# #     max_steps = int(os.environ.get("MAX_STEPS", "20000"))

# #     lr = float(os.environ.get("LR", "3e-4"))
# #     wd = float(os.environ.get("WD", "0.01"))
# #     grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))

# #     # 强对抗: 重点是把 language probe 压下去
# #     # 建议从 5 开始，如果你希望更激进可以 10 或 20
# #     lambda_lang = float(os.environ.get("LAMBDA_LANG", "10.0"))
# #     grl_alpha = float(os.environ.get("GRL_ALPHA", "10.0"))

# #     # 这些正则项用于避免 representation 彻底塌缩，可以按需调小
# #     lambda_align = float(os.environ.get("LAMBDA_ALIGN", "1.0"))
# #     lambda_varcov = float(os.environ.get("LAMBDA_VARCOV", "0.1"))
# #     lambda_mean = float(os.environ.get("LAMBDA_MEAN", "1.0"))
# #     lambda_mean_diff = float(os.environ.get("LAMBDA_MEAN_DIFF", "0.3"))
# #     lambda_pair = float(os.environ.get("LAMBDA_PAIR", "0.01"))

# #     # 语言分类器训练强一点，有助于把 encoder 逼到“更难泄漏语言”
# #     lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "1e-3"))
# #     lang_clf_steps = int(os.environ.get("LANG_CLF_STEPS", "2"))

# #     cmd = [
# #         sys.executable, f.TRAIN_ENTRY,
# #         f.RUN_DIR_FLAG, run_dir,
# #         f.SEED_FLAG, str(seed),
# #         f.BATCH_SIZE_FLAG, str(batch_size),
# #         f.MAX_STEPS_FLAG, str(max_steps),
# #         f.LR_FLAG, str(lr),
# #         f.WD_FLAG, str(wd),
# #         f.GRAD_CLIP_FLAG, str(grad_clip),

# #         f.L_ALIGN_FLAG, str(lambda_align),
# #         f.L_VARCOV_FLAG, str(lambda_varcov),
# #         f.L_MEAN_FLAG, str(lambda_mean),
# #         f.L_MEAN_DIFF_FLAG, str(lambda_mean_diff),
# #         f.L_PAIR_FLAG, str(lambda_pair),

# #         f.L_LANG_FLAG, str(lambda_lang),
# #         f.GRL_ALPHA_FLAG, str(grl_alpha),
# #         f.LANG_CLF_LR_FLAG, str(lang_clf_lr),
# #         f.LANG_CLF_STEPS_FLAG, str(lang_clf_steps),
# #     ]

# #     run(cmd)

# #     print("\n[Next]")
# #     print(f"python src/eval_diag.py --ckpt {run_dir}/ckpt.pt --batch_size 8")
# #     print(f"python src/probe_leakage.py --ckpt {run_dir}/ckpt.pt --max_train_groups 8000 --max_valid_groups 946")


# # if __name__ == "__main__":
# #     main()



# # src/train_planA_strong_adv.py
# import os
# import shlex
# import subprocess
# import sys
# from dataclasses import dataclass


# @dataclass
# class Flags:
#     # 你的训练入口脚本路径
#     TRAIN_ENTRY: str = os.environ.get("TRAIN_ENTRY", "src/train_adv.py")

#     # Flag definitions
#     RUN_DIR_FLAG: str = os.environ.get("RUN_DIR_FLAG", "--run_dir")
#     SEED_FLAG: str = os.environ.get("SEED_FLAG", "--seed")
#     BATCH_SIZE_FLAG: str = os.environ.get("BATCH_SIZE_FLAG", "--batch_size")
#     MAX_STEPS_FLAG: str = os.environ.get("MAX_STEPS_FLAG", "--max_steps")
    
#     # --- FIX BELOW ---
#     # Changed "--lr" to "--lr_model" to avoid ambiguity with --lr_lang
#     LR_FLAG: str = os.environ.get("LR_FLAG", "--lr_model") 
#     # -----------------

#     WD_FLAG: str = os.environ.get("WD_FLAG", "--weight_decay")
#     GRAD_CLIP_FLAG: str = os.environ.get("GRAD_CLIP_FLAG", "--grad_clip")

#     # loss 权重相关 flag
#     L_ALIGN_FLAG: str = os.environ.get("L_ALIGN_FLAG", "--lambda_align")
#     L_VARCOV_FLAG: str = os.environ.get("L_VARCOV_FLAG", "--lambda_varcov")
#     L_MEAN_FLAG: str = os.environ.get("L_MEAN_FLAG", "--lambda_mean")
#     L_MEAN_DIFF_FLAG: str = os.environ.get("L_MEAN_DIFF_FLAG", "--lambda_mean_diff")
#     L_PAIR_FLAG: str = os.environ.get("L_PAIR_FLAG", "--lambda_pair")
#     L_LANG_FLAG: str = os.environ.get("L_LANG_FLAG", "--lambda_lang")

#     # 语言对抗机制相关 flag
#     GRL_ALPHA_FLAG: str = os.environ.get("GRL_ALPHA_FLAG", "--grl_alpha")
#     LANG_CLF_LR_FLAG: str = os.environ.get("LANG_CLF_LR_FLAG", "--lang_clf_lr")
#     LANG_CLF_STEPS_FLAG: str = os.environ.get("LANG_CLF_STEPS_FLAG", "--lang_clf_steps")


# def run(cmd: list[str]) -> None:
#     print("\n[CMD]")
#     print(" ".join(shlex.quote(x) for x in cmd))
#     subprocess.run(cmd, check=True)


# def main():
#     f = Flags()

#     # Environment variable overrides
#     run_dir = os.environ.get("RUN_DIR", "runs/planA_strong_adv")
#     seed = int(os.environ.get("SEED", "0"))
#     batch_size = int(os.environ.get("BATCH_SIZE", "8"))
#     max_steps = int(os.environ.get("MAX_STEPS", "20000"))

#     lr = float(os.environ.get("LR", "3e-4"))
#     wd = float(os.environ.get("WD", "0.01"))
#     grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))

#     # Strong adversarial settings
#     lambda_lang = float(os.environ.get("LAMBDA_LANG", "10.0"))
#     grl_alpha = float(os.environ.get("GRL_ALPHA", "10.0"))

#     # Regularization settings
#     lambda_align = float(os.environ.get("LAMBDA_ALIGN", "1.0"))
#     lambda_varcov = float(os.environ.get("LAMBDA_VARCOV", "0.1"))
#     lambda_mean = float(os.environ.get("LAMBDA_MEAN", "1.0"))
#     lambda_mean_diff = float(os.environ.get("LAMBDA_MEAN_DIFF", "0.3"))
#     lambda_pair = float(os.environ.get("LAMBDA_PAIR", "0.01"))

#     # Language classifier settings
#     lang_clf_lr = float(os.environ.get("LANG_CLF_LR", "1e-3"))
#     lang_clf_steps = int(os.environ.get("LANG_CLF_STEPS", "2"))

#     cmd = [
#         sys.executable, f.TRAIN_ENTRY,
#         f.RUN_DIR_FLAG, run_dir,
#         f.SEED_FLAG, str(seed),
#         f.BATCH_SIZE_FLAG, str(batch_size),
#         f.MAX_STEPS_FLAG, str(max_steps),
#         f.LR_FLAG, str(lr),
#         f.WD_FLAG, str(wd),
#         f.GRAD_CLIP_FLAG, str(grad_clip),

#         f.L_ALIGN_FLAG, str(lambda_align),
#         f.L_VARCOV_FLAG, str(lambda_varcov),
#         f.L_MEAN_FLAG, str(lambda_mean),
#         f.L_MEAN_DIFF_FLAG, str(lambda_mean_diff),
#         f.L_PAIR_FLAG, str(lambda_pair),

#         f.L_LANG_FLAG, str(lambda_lang),
#         f.GRL_ALPHA_FLAG, str(grl_alpha),
#         f.LANG_CLF_LR_FLAG, str(lang_clf_lr),
#         f.LANG_CLF_STEPS_FLAG, str(lang_clf_steps),
#     ]

#     run(cmd)

#     print("\n[Next]")
#     print(f"python src/eval_diag.py --ckpt {run_dir}/ckpt.pt --batch_size 8")
#     print(f"python src/probe_leakage.py --ckpt {run_dir}/ckpt.pt --max_train_groups 8000 --max_valid_groups 946")


# if __name__ == "__main__":
#     main()


# src/train_planA_strong_adv.py
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Flags:
    # 确保环境变量正确指向 src/train_adv4.py
    TRAIN_ENTRY: str = os.environ.get("TRAIN_ENTRY", "src/train_adv4.py")

    # 定义 train_adv4.py 支持的 Flag
    RUN_DIR_FLAG: str = os.environ.get("RUN_DIR_FLAG", "--run_dir")
    BATCH_SIZE_FLAG: str = os.environ.get("BATCH_SIZE_FLAG", "--batch_size")
    LR_FLAG: str = os.environ.get("LR_FLAG", "--lr_model")
    EPOCHS_FLAG: str = os.environ.get("EPOCHS_FLAG", "--epochs")

    # Loss 权重相关
    L_ALIGN_FLAG: str = os.environ.get("L_ALIGN_FLAG", "--lambda_align")
    L_VARCOV_FLAG: str = os.environ.get("L_VARCOV_FLAG", "--lambda_varcov")
    L_MEAN_FLAG: str = os.environ.get("L_MEAN_FLAG", "--lambda_mean")
    L_MEAN_DIFF_FLAG: str = os.environ.get("L_MEAN_DIFF_FLAG", "--lambda_mean_diff")
    L_PAIR_FLAG: str = os.environ.get("L_PAIR_FLAG", "--lambda_pair")
    L_LANG_FLAG: str = os.environ.get("L_LANG_FLAG", "--lambda_lang")

    # 对抗相关
    GRL_ALPHA_FLAG: str = os.environ.get("GRL_ALPHA_FLAG", "--grl_alpha")
    LANG_CLF_LR_FLAG: str = os.environ.get("LANG_CLF_LR_FLAG", "--lr_lang")


def run(cmd: list[str]) -> None:
    print("\n[CMD]")
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main():
    f = Flags()

    # 配置参数
    run_dir = os.environ.get("RUN_DIR", "runs/planA_strong_adv")
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    epochs = int(os.environ.get("EPOCHS", "5")) # train_adv4.py 使用 epochs
    
    lr_model = float(os.environ.get("LR", "3e-4"))
    lr_lang = float(os.environ.get("LANG_CLF_LR", "1e-3"))

    # 权重设置
    lambda_lang = float(os.environ.get("LAMBDA_LANG", "10.0"))
    grl_alpha = float(os.environ.get("GRL_ALPHA", "10.0"))
    lambda_align = float(os.environ.get("LAMBDA_ALIGN", "1.0"))
    lambda_varcov = float(os.environ.get("LAMBDA_VARCOV", "0.1"))
    lambda_mean = float(os.environ.get("LAMBDA_MEAN", "1.0"))
    lambda_mean_diff = float(os.environ.get("LAMBDA_MEAN_DIFF", "0.3"))
    lambda_pair = float(os.environ.get("LAMBDA_PAIR", "0.01"))

    # 构造命令 (只包含 train_adv4.py argparse 存在的参数)
    cmd = [
        sys.executable, f.TRAIN_ENTRY,
        f.RUN_DIR_FLAG, run_dir,
        f.BATCH_SIZE_FLAG, str(batch_size),
        f.EPOCHS_FLAG, str(epochs),
        f.LR_FLAG, str(lr_model),
        f.LANG_CLF_LR_FLAG, str(lr_lang),
        f.L_ALIGN_FLAG, str(lambda_align),
        f.L_VARCOV_FLAG, str(lambda_varcov),
        f.L_MEAN_FLAG, str(lambda_mean),
        f.L_MEAN_DIFF_FLAG, str(lambda_mean_diff),
        f.L_PAIR_FLAG, str(lambda_pair),
        f.L_LANG_FLAG, str(lambda_lang),
        f.GRL_ALPHA_FLAG, str(grl_alpha),
    ]

    run(cmd)

    # 打印后续步骤
    print("\n[Next Steps]")
    print(f"python src/eval_diag.py --ckpt {run_dir}/ckpt.pt --batch_size 8")
    print(f"python src/probe_leakage.py --ckpt {run_dir}/ckpt.pt --max_train_groups 8000 --max_valid_groups 946")


if __name__ == "__main__":
    main()

