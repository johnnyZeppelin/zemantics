# src/models/lang_clf.py
from __future__ import annotations
import torch
import torch.nn as nn


class LangClfMLP(nn.Module):
    """
    2-layer MLP language classifier for adversarial training.
    Input: z (B, D)
    Output: logits (B, 2) for {en, zh}
    """
    def __init__(self, d: int = 512, h: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
