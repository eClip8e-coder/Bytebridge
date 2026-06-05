from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BoundaryByteAdapterConfig:
    byte_vocab_size: int = 260
    byte_embed_dim: int = 128
    hidden_dim: int = 256
    llm_hidden_size: int = 896
    max_bytes: int = 128
    max_spans: int = 32
    depth: int = 2
    dropout: float = 0.05


class BoundaryByteAdapter(nn.Module):
    pad_id = 256

    def __init__(self, config: BoundaryByteAdapterConfig):
        super().__init__()
        self.config = config
        self.byte_embed = nn.Embedding(config.byte_vocab_size, config.byte_embed_dim, padding_idx=self.pad_id)
        layers: list[nn.Module] = [
            nn.Linear(config.byte_embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        ]
        for _ in range(config.depth):
            layers.append(
                nn.Sequential(
                    nn.LayerNorm(config.hidden_dim),
                    nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                )
            )
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.llm_hidden_size))

    def forward(
        self,
        byte_ids: torch.Tensor,
        span_bounds: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        byte_embeds = self.byte_embed(byte_ids)
        batch, max_spans, _ = span_bounds.shape
        pooled = byte_embeds.new_zeros((batch, max_spans, byte_embeds.shape[-1]))
        for b in range(batch):
            for s in range(max_spans):
                if not bool(span_mask[b, s]):
                    continue
                start = int(span_bounds[b, s, 0].item())
                end = int(span_bounds[b, s, 1].item())
                start = max(0, min(start, byte_embeds.shape[1] - 1))
                end = max(start + 1, min(end, byte_embeds.shape[1]))
                pooled[b, s] = byte_embeds[b, start:end].mean(dim=0)
        hidden = self.encoder(pooled)
        return self.proj(hidden), span_mask
