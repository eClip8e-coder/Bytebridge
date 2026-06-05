from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F


def causal_lm_loss_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample causal LM loss and valid shifted-token counts."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    token_loss = flat_loss.view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    denom = valid.sum(dim=1).clamp_min(1)
    per_sample = (token_loss * valid).sum(dim=1) / denom
    return per_sample, valid.sum(dim=1)


def safe_ppl(loss: float) -> float:
    if loss > 20:
        return float("inf")
    return float(math.exp(loss))


def aggregate_by_bucket(rows: list[dict], clean_bucket: str = "clean_english") -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["bucket"]].append(row)

    out = {}
    clean_loss = None
    if clean_bucket in grouped:
        clean_loss = sum(r["loss"] for r in grouped[clean_bucket]) / len(grouped[clean_bucket])

    clean_by_id = {
        r["clean_id"]: r["loss"]
        for r in grouped.get(clean_bucket, [])
        if r.get("clean_id")
    }

    for bucket, bucket_rows in sorted(grouped.items()):
        mean_loss = sum(r["loss"] for r in bucket_rows) / len(bucket_rows)
        token_count = sum(r.get("token_count", 0) for r in bucket_rows)
        byte_count = sum(r.get("byte_count", 0) for r in bucket_rows)
        paired_deltas = [
            r["loss"] - clean_by_id[r["clean_id"]]
            for r in bucket_rows
            if r.get("clean_id") in clean_by_id and bucket != clean_bucket
        ]
        out[bucket] = {
            "count": len(bucket_rows),
            "loss": mean_loss,
            "ppl": safe_ppl(mean_loss),
            "mean_token_count": token_count / max(1, len(bucket_rows)),
            "mean_byte_count": byte_count / max(1, len(bucket_rows)),
            "degradation_ratio_vs_clean_bucket": None if clean_loss is None else mean_loss / clean_loss,
            "paired_loss_delta_vs_clean": None
            if not paired_deltas
            else sum(paired_deltas) / len(paired_deltas),
        }
    return out
