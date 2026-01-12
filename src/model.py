from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSeq2SeqLM


@dataclass
class ModelOutputs:
    # logits for en_doc -> en_sum
    logits_en: torch.Tensor  # [B, L, V]
    # logits for zh_doc -> en_sum
    logits_zh: torch.Tensor  # [B, L, V]
    # pooled latent for alignment
    zbar_en: torch.Tensor    # [B, D]
    zbar_zh: torch.Tensor    # [B, D]
    # full latent slots (optional, for diagnostics)
    z_en: torch.Tensor       # [B, K, D]
    z_zh: torch.Tensor       # [B, K, D]


class LatentBottleneck(nn.Module):
    """
    Perceiver-style bottleneck:
      encoder hidden states H: [B, T, D]
      learnable queries Q: [K, D]
      output latent slots Z: [B, K, D]
    Implemented with nn.MultiheadAttention in "batch_first" mode.
    """

    def __init__(
        self,
        d_model: int,
        num_latents: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
        add_self_attn: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.add_self_attn = add_self_attn

        self.latent_queries = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_ln = nn.LayerNorm(d_model)
        self.cross_ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ff_ln = nn.LayerNorm(d_model)

        if add_self_attn:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.self_ln = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, h_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        h: [B, T, D]
        h_attn_mask: [B, T] where 1 means keep, 0 means pad (HF style)
        returns Z: [B, K, D]
        """
        bsz = h.size(0)

        # Build queries: [B, K, D]
        q = self.latent_queries.unsqueeze(0).expand(bsz, -1, -1)

        # MultiheadAttention key_padding_mask expects True for PAD positions.
        key_padding_mask = None
        if h_attn_mask is not None:
            key_padding_mask = (h_attn_mask == 0)  # [B, T] bool

        # Cross-attn: queries attend to encoder hidden states
        z, _ = self.cross_attn(query=q, key=h, value=h, key_padding_mask=key_padding_mask, need_weights=False)

        # Residual + LN + FF
        z = self.cross_ln(q + self.dropout(z))
        z2 = self.cross_ff(z)
        z = self.ff_ln(z + z2)

        # Optional: self-attn over latents
        if self.add_self_attn:
            z_sa, _ = self.self_attn(query=z, key=z, value=z, need_weights=False)
            z = self.self_ln(z + self.dropout(z_sa))

        return z


class LatentRendererModel(nn.Module):
    """
    Wraps a pretrained seq2seq LM (mT5-small) and inserts a bottleneck between encoder and decoder.
    Decoder cross-attends only to Z (latent slots), not to full encoder states.
    """

    def __init__(
        self,
        backbone_name: str = "google/mt5-small",
        num_latents: int = 16,
        bottleneck_heads: int = 8,
        latent_dropout: float = 0.1,
        latent_noise_std: float = 0.01,
        add_latent_self_attn: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(backbone_name)
        self.config = self.model.config

        d_model = self.config.d_model
        # Safeguard: choose heads <= d_model and divisible
        if d_model % bottleneck_heads != 0:
            # fallback to a divisor of d_model
            for h in [8, 4, 2, 1]:
                if d_model % h == 0:
                    bottleneck_heads = h
                    break

        self.bottleneck = LatentBottleneck(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=bottleneck_heads,
            dropout=latent_dropout,
            add_self_attn=add_latent_self_attn,
        )
        self.latent_dropout = nn.Dropout(latent_dropout)
        self.latent_noise_std = latent_noise_std

    @torch.no_grad()
    def get_vocab_size(self) -> int:
        return self.config.vocab_size

    def encode_to_z(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        attention_mask: [B, T]
        returns Z: [B, K, D]
        """
        enc_out = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        h = enc_out.last_hidden_state  # [B, T, D]
        z = self.bottleneck(h, h_attn_mask=attention_mask)  # [B, K, D]
        z = self.latent_dropout(z)

        if self.training and add_noise and self.latent_noise_std > 0:
            z = z + torch.randn_like(z) * self.latent_noise_std

        return z

    def decode_from_z(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        z: [B, K, D]
        labels: [B, L] with -100 masked pads (HF convention)
        returns logits: [B, L, V]
        """
        # Create a fake encoder_outputs object with last_hidden_state = z
        # and an attention mask of ones (no padding among latents).
        bsz, k, _ = z.shape
        z_attn = torch.ones((bsz, k), dtype=torch.long, device=z.device)

        out = self.model(
            input_ids=None,
            # attention_mask=None,
            encoder_outputs=(z,),  # tuple form: last_hidden_state
            # encoder_attention_mask=z_attn,
            attention_mask=z_attn,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return out.logits  # [B, L, V]

    @staticmethod
    def pool_zbar(z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, K, D] -> zbar: [B, D], mean pooling then L2 normalize.
        """
        zbar = z.mean(dim=1)
        zbar = F.normalize(zbar, p=2, dim=-1)
        return zbar

    def forward(
        self,
        en_input_ids: torch.Tensor,
        en_attention_mask: torch.Tensor,
        zh_input_ids: torch.Tensor,
        zh_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> ModelOutputs:
        # Encode both views into latent slots
        z_en = self.encode_to_z(en_input_ids, en_attention_mask, add_noise=True)
        z_zh = self.encode_to_z(zh_input_ids, zh_attention_mask, add_noise=True)

        # Decode both into the SAME target space (en_sum labels)
        logits_en = self.decode_from_z(z_en, labels=labels)
        logits_zh = self.decode_from_z(z_zh, labels=labels)

        # pooled vectors for alignment
        zbar_en = self.pool_zbar(z_en)
        zbar_zh = self.pool_zbar(z_zh)

        return ModelOutputs(
            logits_en=logits_en,
            logits_zh=logits_zh,
            zbar_en=zbar_en,
            zbar_zh=zbar_zh,
            z_en=z_en,
            z_zh=z_zh,
        )
