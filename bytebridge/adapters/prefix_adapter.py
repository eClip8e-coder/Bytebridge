from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BytePrefixAdapterConfig:
    byte_vocab_size: int = 260
    byte_embed_dim: int = 128
    hidden_dim: int = 256
    llm_hidden_size: int = 896
    max_bytes: int = 128
    prefix_length: int = 32
    depth: int = 2
    dropout: float = 0.05
    pooling_mode: str = "fixed"


class BytePrefixAdapter(nn.Module):
    pad_id = 256

    def __init__(self, config: BytePrefixAdapterConfig):
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
        self.byte_encoder = nn.Sequential(*layers)
        self.prefix_pos = nn.Parameter(torch.zeros(config.prefix_length, config.hidden_dim))
        nn.init.normal_(self.prefix_pos, std=0.02)
        if config.pooling_mode == "attention":
            self.query = nn.Parameter(torch.randn(config.prefix_length, config.hidden_dim) * 0.02)
            self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.value = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.proj = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.llm_hidden_size))

    def fixed_pool(self, hidden: torch.Tensor, byte_mask: torch.Tensor) -> torch.Tensor:
        batch, byte_len, dim = hidden.shape
        prefix = hidden.new_zeros((batch, self.config.prefix_length, dim))
        for p in range(self.config.prefix_length):
            start = p * byte_len // self.config.prefix_length
            end = (p + 1) * byte_len // self.config.prefix_length
            end = max(start + 1, end)
            span_hidden = hidden[:, start:end]
            span_mask = byte_mask[:, start:end].float()
            denom = span_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            prefix[:, p] = (span_hidden * span_mask.unsqueeze(-1)).sum(dim=1) / denom
        return prefix

    def mean_pool(self, hidden: torch.Tensor, byte_mask: torch.Tensor) -> torch.Tensor:
        denom = byte_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (hidden * byte_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return pooled.unsqueeze(1).expand(-1, self.config.prefix_length, -1)

    def attention_pool(self, hidden: torch.Tensor, byte_mask: torch.Tensor) -> torch.Tensor:
        q = self.query.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        k = self.key(hidden)
        v = self.value(hidden)
        scores = torch.matmul(q, k.transpose(1, 2)) / (hidden.shape[-1] ** 0.5)
        scores = scores.masked_fill(~byte_mask.unsqueeze(1), -1e4)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        byte_mask = byte_ids.ne(self.pad_id)
        hidden = self.byte_encoder(self.byte_embed(byte_ids))
        if self.config.pooling_mode == "fixed":
            prefix = self.fixed_pool(hidden, byte_mask)
        elif self.config.pooling_mode == "mean":
            prefix = self.mean_pool(hidden, byte_mask)
        elif self.config.pooling_mode == "attention":
            prefix = self.attention_pool(hidden, byte_mask)
        else:
            raise ValueError(f"Unknown pooling_mode: {self.config.pooling_mode}")
        prefix = prefix + self.prefix_pos.unsqueeze(0)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=torch.bool, device=prefix.device)
        return self.proj(prefix), prefix_mask
