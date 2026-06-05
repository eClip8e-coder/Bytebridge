from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from bytebridge.adapters import ByteAdapter, ByteAdapterConfig
from bytebridge.data import encode_bytes_batch, read_jsonl
from bytebridge.utils import seed_everything
from scripts.phase2_train_bytebridge import checksum_first_tensors, filter_rows, grad_checksum


def collate_distill(batch: list[dict], tokenizer, max_bytes: int, max_token_length: int, prompt_suffix: str) -> dict:
    texts = [sample["input_text"] + prompt_suffix for sample in batch]
    byte_ids = encode_bytes_batch(texts, max_bytes)
    token_rows = []
    token_lengths = []
    token_truncated = []
    for sample in batch:
        ids = tokenizer(sample["input_text"] + prompt_suffix, add_special_tokens=False, truncation=False)["input_ids"]
        raw_len = len(ids)
        ids = ids[:max_token_length]
        token_rows.append(torch.tensor(ids, dtype=torch.long))
        token_lengths.append(len(ids))
        token_truncated.append(raw_len > len(ids))
    token_ids = pad_sequence(token_rows, batch_first=True, padding_value=tokenizer.pad_token_id)
    token_mask = token_ids.ne(tokenizer.pad_token_id)
    return {
        "samples": batch,
        "byte_ids": byte_ids,
        "token_ids": token_ids,
        "token_mask": token_mask,
        "token_lengths": token_lengths,
        "token_truncated": token_truncated,
    }


def token_span_targets(token_embeds: torch.Tensor, token_mask: torch.Tensor, num_latents: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool variable token sequences into num_latents deterministic spans."""
    batch, _, hidden = token_embeds.shape
    targets = token_embeds.new_zeros((batch, num_latents, hidden))
    target_mask = torch.zeros((batch, num_latents), dtype=torch.bool, device=token_embeds.device)
    lengths = token_mask.sum(dim=1).tolist()
    for i, length in enumerate(lengths):
        t = int(length)
        if t <= 0:
            continue
        valid = token_embeds[i, :t]
        for j in range(num_latents):
            start = int(j * t // num_latents)
            end = int((j + 1) * t // num_latents)
            if end <= start:
                start = min(start, t - 1)
                end = min(start + 1, t)
            targets[i, j] = valid[start:end].float().mean(dim=0)
            target_mask[i, j] = True
    return targets, target_mask


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred.float() - target.float()).pow(2).mean(dim=-1)
    denom = mask.sum().clamp_min(1)
    return (diff * mask.float()).sum() / denom


def masked_cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(pred.float(), target.float(), dim=-1)
    denom = mask.sum().clamp_min(1)
    return ((1.0 - cos) * mask.float()).sum() / denom


def pooled_loss(byte_latents: torch.Tensor, byte_mask: torch.Tensor, token_embeds: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    byte_denom = byte_mask.sum(dim=1, keepdim=True).clamp_min(1)
    token_denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1)
    byte_mean = (byte_latents.float() * byte_mask.unsqueeze(-1).float()).sum(dim=1) / byte_denom
    token_mean = (token_embeds.float() * token_mask.unsqueeze(-1).float()).sum(dim=1) / token_denom
    return F.mse_loss(byte_mean, token_mean) + (1.0 - F.cosine_similarity(byte_mean, token_mean, dim=-1)).mean()


def env(model_name: str, cfg: dict) -> dict:
    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "torch_cuda_version": torch.version.cuda,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "model_name": model_name,
        "config": cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "config.yaml")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    initial_checksum = checksum_first_tensors(llm)
    embedding = llm.get_input_embeddings()

    hidden_size = embedding.embedding_dim
    adapter_cfg = ByteAdapterConfig(
        byte_embed_dim=int(cfg["byte_embed_dim"]),
        hidden_dim=int(cfg["adapter_hidden_dim"]),
        llm_hidden_size=hidden_size,
        patch_size=int(cfg["patch_size"]),
        max_bytes=int(cfg["max_prompt_bytes"]),
        depth=int(cfg["adapter_depth"]),
        dropout=float(cfg["dropout"]),
    )
    adapter = ByteAdapter(adapter_cfg).to(device)
    adapter.train()

    train_rows = filter_rows(read_jsonl(cfg["train_manifest_path"]), list(cfg["train_buckets"]))
    loader = DataLoader(
        train_rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: collate_distill(
            b,
            tokenizer,
            int(cfg["max_prompt_bytes"]),
            int(cfg["max_token_length"]),
            str(cfg.get("prompt_suffix", "\n")),
        ),
    )
    optimizer = AdamW(adapter.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    metrics_path = out_dir / "train_metrics.csv"
    train_iter = iter(loader)
    first_loss = None
    last_row = None
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "fixed_span_mse",
                "fixed_span_cosine",
                "pooled_loss",
                "adapter_grad_abs_sum",
                "llm_grad_abs_sum",
                "gpu_mem_allocated_mb",
                "gpu_mem_reserved_mb",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        for step in range(1, int(cfg["steps"]) + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)

            byte_ids = batch["byte_ids"].to(device)
            token_ids = batch["token_ids"].to(device)
            token_mask = batch["token_mask"].to(device)
            byte_latents, byte_mask = adapter(byte_ids)
            safe_token_ids = token_ids.masked_fill(~token_mask, tokenizer.pad_token_id)
            with torch.no_grad():
                token_embeds = embedding(safe_token_ids).detach()
            span_targets, span_mask = token_span_targets(token_embeds, token_mask, byte_latents.shape[1])
            valid_span_mask = span_mask & byte_mask.to(device)
            span_mse = masked_mse(byte_latents, span_targets, valid_span_mask)
            span_cos = masked_cosine_loss(byte_latents, span_targets, valid_span_mask)
            seq_loss = pooled_loss(byte_latents, byte_mask.to(device), token_embeds, token_mask)
            loss = (
                float(cfg["span_mse_weight"]) * span_mse
                + float(cfg["span_cosine_weight"]) * span_cos
                + float(cfg["pooled_weight"]) * seq_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            adapter_grad = grad_checksum(adapter.parameters())
            llm_grad = grad_checksum(llm.parameters())
            optimizer.step()

            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "fixed_span_mse": float(span_mse.detach().cpu()),
                "fixed_span_cosine": float(span_cos.detach().cpu()),
                "pooled_loss": float(seq_loss.detach().cpu()),
                "adapter_grad_abs_sum": adapter_grad,
                "llm_grad_abs_sum": llm_grad,
                "gpu_mem_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0,
                "gpu_mem_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0,
                "elapsed_sec": time.perf_counter() - start,
            }
            if first_loss is None:
                first_loss = row["loss"]
            last_row = row
            writer.writerow(row)
            if step == 1 or step % int(cfg["log_every"]) == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)

    final_checksum = checksum_first_tensors(llm)
    summary = {
        "run_name": cfg["run_name"],
        "model_name": cfg["model_name"],
        "steps": int(cfg["steps"]),
        "train_rows": len(train_rows),
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "llm_trainable_params": sum(p.numel() for p in llm.parameters() if p.requires_grad),
        "llm_checksum_delta_first8_tensors": final_checksum - initial_checksum,
        "initial_loss": first_loss,
        "final_loss": last_row["loss"] if last_row else None,
        "loss_delta": None if first_loss is None or last_row is None else first_loss - last_row["loss"],
        "final_fixed_span_mse": None if last_row is None else last_row["fixed_span_mse"],
        "final_fixed_span_cosine": None if last_row is None else last_row["fixed_span_cosine"],
        "final_pooled_loss": None if last_row is None else last_row["pooled_loss"],
        "gpu_memory_allocated_peak_mib": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0,
        "gpu_memory_reserved_peak_mib": torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0,
        "total_elapsed_sec": time.perf_counter() - start,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env(cfg["model_name"], cfg), f, indent=2, ensure_ascii=False)
    ckpt = {
        "adapter_state_dict": adapter.state_dict(),
        "adapter_config": adapter_cfg.__dict__,
        "model_name": cfg["model_name"],
        "step": int(cfg["steps"]),
        "distill_summary": summary,
    }
    torch.save(ckpt, out_dir / "adapter.pt")
    torch.save(ckpt, out_dir / "final_adapter.pt")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
