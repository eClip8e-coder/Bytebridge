from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.cache_utils import DynamicCache


@dataclass
class ByteKVPrefixAdapterConfig:
    byte_vocab_size: int = 260
    byte_embed_dim: int = 128
    hidden_dim: int = 256
    max_bytes: int = 128
    prefix_length: int = 32
    depth: int = 2
    dropout: float = 0.05
    pooling_mode: str = "fixed"
    num_hidden_layers: int = 24
    num_layers_injected: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 64
    target_layers: str = "first_n"


class ByteKVPrefixAdapter(nn.Module):
    pad_id = 256

    def __init__(self, config: ByteKVPrefixAdapterConfig):
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
        self.prefix_pos = nn.Parameter(torch.randn(config.prefix_length, config.hidden_dim) * 0.02)
        if config.pooling_mode == "attention":
            self.query = nn.Parameter(torch.randn(config.prefix_length, config.hidden_dim) * 0.02)
            self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.value = nn.Linear(config.hidden_dim, config.hidden_dim)
        injected = self.injected_layer_indices()
        out_dim = len(injected) * 2 * config.num_key_value_heads * config.head_dim
        self.to_kv = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, out_dim))

    def injected_layer_indices(self) -> list[int]:
        if self.config.target_layers == "all":
            return list(range(self.config.num_hidden_layers))
        if self.config.target_layers == "first_n":
            return list(range(min(self.config.num_layers_injected, self.config.num_hidden_layers)))
        raise ValueError(f"Unsupported target_layers: {self.config.target_layers}")

    def fixed_pool(self, hidden: torch.Tensor, byte_mask: torch.Tensor) -> torch.Tensor:
        batch, byte_len, dim = hidden.shape
        prefix = hidden.new_zeros((batch, self.config.prefix_length, dim))
        for p in range(self.config.prefix_length):
            start = p * byte_len // self.config.prefix_length
            end = max(start + 1, (p + 1) * byte_len // self.config.prefix_length)
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

    def prefix_hidden(self, byte_ids: torch.Tensor) -> torch.Tensor:
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
        return prefix + self.prefix_pos.unsqueeze(0)

    def forward(self, byte_ids: torch.Tensor, model_config, dtype: torch.dtype) -> tuple[DynamicCache, torch.Tensor, list[int]]:
        batch = byte_ids.shape[0]
        device = byte_ids.device
        prefix = self.prefix_hidden(byte_ids)
        injected = self.injected_layer_indices()
        raw = self.to_kv(prefix)
        raw = raw.view(
            batch,
            self.config.prefix_length,
            len(injected),
            2,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        raw = raw.permute(2, 3, 0, 4, 1, 5).contiguous()
        cache = DynamicCache(config=model_config)
        trainable_by_layer = {layer_idx: i for i, layer_idx in enumerate(injected)}
        for layer_idx in range(self.config.num_hidden_layers):
            if layer_idx in trainable_by_layer:
                offset = trainable_by_layer[layer_idx]
                key = raw[offset, 0].to(dtype=dtype)
                value = raw[offset, 1].to(dtype=dtype)
            else:
                key = torch.zeros(
                    batch,
                    self.config.num_key_value_heads,
                    self.config.prefix_length,
                    self.config.head_dim,
                    dtype=dtype,
                    device=device,
                )
                value = torch.zeros_like(key)
            cache.update(key, value, layer_idx)
        prefix_mask = torch.ones((batch, self.config.prefix_length), dtype=torch.bool, device=device)
        return cache, prefix_mask, injected
