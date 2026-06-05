from __future__ import annotations

from typing import Iterable

import torch


def build_toy_corpus() -> list[str]:
    return [
        "ByteBridge learns from clean text.",
        "Frozen language models accept input embeddings.",
        "Unicode text includes cafe, naive, and emoji :)",
        "JSON uses braces, quotes, numbers, and commas.",
        "Python code: value = items[0] + 42",
        "Small adapters can overfit tiny experiments.",
        "Tokenizer-free inputs start as UTF-8 bytes.",
        "Research prototypes need honest negative results.",
    ]


def encode_bytes_batch(texts: Iterable[str], max_bytes: int, pad_id: int = 256) -> torch.Tensor:
    rows: list[list[int]] = []
    for text in texts:
        raw = list(text.encode("utf-8", errors="replace"))[:max_bytes]
        rows.append(raw + [pad_id] * (max_bytes - len(raw)))
    return torch.tensor(rows, dtype=torch.long)
