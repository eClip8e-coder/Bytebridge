from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ByteAdapterConfig:
    byte_vocab_size: int = 260
    byte_embed_dim: int = 128
    hidden_dim: int = 256
    llm_hidden_size: int = 896
    patch_size: int = 4
    max_bytes: int = 128
    depth: int = 2
    dropout: float = 0.05


class ByteAdapter(nn.Module):
    """Minimal fixed-patch byte-to-latent adapter.

    Byte ids use 0..255 for byte values, 256 for padding, and 257 for invalid
    bytes if a future dataset needs lossy decoding. The first prototype uses
    fixed-size non-overlapping byte patches and projects one latent per patch.
    """

    pad_id = 256
    invalid_id = 257

    def __init__(self, config: ByteAdapterConfig):
        super().__init__()
        self.config = config
        self.byte_embed = nn.Embedding(
            config.byte_vocab_size,
            config.byte_embed_dim,
            padding_idx=self.pad_id,
        )
        self.local_encoder = nn.Sequential(
            nn.Linear(config.patch_size * config.byte_embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            *[
                nn.Sequential(
                    nn.LayerNorm(config.hidden_dim),
                    nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                )
                for _ in range(config.depth)
            ],
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.llm_hidden_size),
        )

    @property
    def num_patches(self) -> int:
        return (self.config.max_bytes + self.config.patch_size - 1) // self.config.patch_size

    def forward(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input embeddings and patch attention mask.

        Args:
            byte_ids: Long tensor of shape [batch, max_bytes].

        Returns:
            embeds: Float tensor [batch, num_patches, llm_hidden_size].
            patch_mask: Bool tensor [batch, num_patches] where true means the
                patch contains at least one non-padding byte.
        """
        batch, byte_len = byte_ids.shape
        pad_len = self.num_patches * self.config.patch_size - byte_len
        if pad_len > 0:
            pad = torch.full(
                (batch, pad_len),
                self.pad_id,
                dtype=byte_ids.dtype,
                device=byte_ids.device,
            )
            byte_ids = torch.cat([byte_ids, pad], dim=1)

        byte_embeds = self.byte_embed(byte_ids)
        patches = byte_embeds.view(batch, self.num_patches, -1)
        patch_mask = byte_ids.view(batch, self.num_patches, self.config.patch_size).ne(self.pad_id).any(dim=-1)
        hidden = self.local_encoder(patches)
        return self.proj(hidden), patch_mask
