# src/adv/adv_lang.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn

from src.utils.grad_reverse import grad_reverse


@dataclass
class AdvLangConfig:
    lambda_lang: float = 2.0     # weight on encoder adversarial loss
    grl_alpha: float = 1.0       # GRL strength
    k_lang: int = 5              # critic steps per iteration
    detach_z_for_clf: bool = True


@torch.no_grad()
def lang_acc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def update_lang_critic(
    z: torch.Tensor,
    lang_id: torch.Tensor,
    lang_clf: nn.Module,
    opt_lang: torch.optim.Optimizer,
    k_lang: int = 5,
) -> dict:
    """
    Train language classifier to predict lang_id from z.
    z: (B, D)
    lang_id: (B,) int64 in {0,1}
    """
    lang_clf.train()
    stats = {}
    for i in range(k_lang):
        opt_lang.zero_grad(set_to_none=True)
        logits = lang_clf(z.detach())
        loss = F.cross_entropy(logits, lang_id)
        loss.backward()
        opt_lang.step()

        if i == k_lang - 1:
            stats["lang_loss_clf"] = float(loss.item())
            stats["lang_acc_clf_detached"] = float(lang_acc_from_logits(logits, lang_id))
    return stats


def encoder_adv_lang_loss(
    z: torch.Tensor,
    lang_id: torch.Tensor,
    lang_clf: nn.Module,
    grl_alpha: float = 1.0,
) -> torch.Tensor:
    """
    Adversarial loss for encoder: use GRL to push z to be language-invariant.
    """
    lang_clf.eval()  # optional: you can keep train() too, but eval is stabler with LN
    z_grl = grad_reverse(z, alpha=grl_alpha)
    logits = lang_clf(z_grl)
    return F.cross_entropy(logits, lang_id)
