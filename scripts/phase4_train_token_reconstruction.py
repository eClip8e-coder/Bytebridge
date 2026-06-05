from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from bytebridge.adapters import BoundaryByteAdapter, BoundaryByteAdapterConfig
from bytebridge.data import encode_bytes_batch, read_jsonl, write_jsonl
from bytebridge.data.token_byte_mapping import (
    fixed_byte_spans,
    heuristic_byte_spans,
    label_spans_by_overlap,
    token_byte_spans,
)
from bytebridge.utils import seed_everything
from scripts.phase2_train_bytebridge import checksum_first_tensors, filter_rows, grad_checksum


def make_spans_and_labels(text: str, tokenizer, mode: str, max_spans: int, max_bytes: int, fixed_patch_size: int, heuristic_max_span_bytes: int):
    token_spans, meta = token_byte_spans(text, tokenizer)
    byte_len = len(text.encode("utf-8"))
    if mode == "oracle":
        mapped = [s for s in token_spans if s.mapped and s.byte_end > s.byte_start]
        spans = [(s.byte_start, min(s.byte_end, max_bytes)) for s in mapped if s.byte_start < max_bytes][:max_spans]
        labels = [s.token_id for s in mapped if s.byte_start < max_bytes][:max_spans]
    elif mode == "heuristic":
        spans = heuristic_byte_spans(text, max_spans, max_bytes, heuristic_max_span_bytes)
        labels = label_spans_by_overlap(spans, token_spans)
    elif mode == "fixed":
        spans = fixed_byte_spans(byte_len, fixed_patch_size, max_spans, max_bytes)
        labels = label_spans_by_overlap(spans, token_spans)
    else:
        raise ValueError(f"Unknown boundary_mode: {mode}")
    return spans, labels, meta


def collate_recon(batch, tokenizer, cfg):
    texts = [sample["input_text"] + str(cfg.get("prompt_suffix", "\n")) for sample in batch]
    max_bytes = int(cfg["max_prompt_bytes"])
    max_spans = int(cfg["max_spans"])
    byte_ids = encode_bytes_batch(texts, max_bytes)
    span_bounds = torch.zeros((len(batch), max_spans, 2), dtype=torch.long)
    span_mask = torch.zeros((len(batch), max_spans), dtype=torch.bool)
    labels = torch.full((len(batch), max_spans), -100, dtype=torch.long)
    metas = []
    for i, text in enumerate(texts):
        spans, span_labels, meta = make_spans_and_labels(
            text,
            tokenizer,
            str(cfg["boundary_mode"]),
            max_spans,
            max_bytes,
            int(cfg["fixed_patch_size"]),
            int(cfg["heuristic_max_span_bytes"]),
        )
        metas.append(meta)
        for j, (start, end) in enumerate(spans[:max_spans]):
            if end <= start:
                continue
            span_bounds[i, j, 0] = start
            span_bounds[i, j, 1] = end
            span_mask[i, j] = True
            labels[i, j] = span_labels[j] if j < len(span_labels) else -100
    return {"samples": batch, "byte_ids": byte_ids, "span_bounds": span_bounds, "span_mask": span_mask, "labels": labels, "metas": metas}


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, int]:
    valid = labels.ne(-100)
    n = int(valid.sum().item())
    if n == 0:
        return 0.0, 0.0, 0
    top1 = logits.argmax(dim=-1)
    top5 = logits.topk(k=5, dim=-1).indices
    acc1 = (top1.eq(labels) & valid).sum().item() / n
    acc5 = ((top5.eq(labels.unsqueeze(-1))).any(dim=-1) & valid).sum().item() / n
    return acc1, acc5, n


def evaluate(adapter, classifier, tokenizer, rows, cfg, device, out_dir, split):
    adapter.eval()
    classifier.eval()
    loader = DataLoader(rows, batch_size=int(cfg["eval_batch_size"]), shuffle=False, collate_fn=lambda b: collate_recon(b, tokenizer, cfg))
    total_loss = 0.0
    total_valid = 0
    bucket_stats = defaultdict(lambda: {"loss_sum": 0.0, "valid": 0, "top1": 0.0, "top5": 0.0, "samples": 0})
    examples = []
    with torch.no_grad():
        for batch in loader:
            latents, mask = adapter(batch["byte_ids"].to(device), batch["span_bounds"].to(device), batch["span_mask"].to(device))
            logits = classifier(latents)
            labels = batch["labels"].to(device)
            loss_flat = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100, reduction="none").view(labels.shape)
            valid = labels.ne(-100)
            for i, sample in enumerate(batch["samples"]):
                n = int(valid[i].sum().item())
                if n == 0:
                    continue
                sample_loss = float(loss_flat[i][valid[i]].mean().cpu())
                a1, a5, _ = accuracy(logits[i : i + 1], labels[i : i + 1])
                bucket = sample["bucket"]
                bucket_stats[bucket]["loss_sum"] += sample_loss * n
                bucket_stats[bucket]["valid"] += n
                bucket_stats[bucket]["top1"] += a1 * n
                bucket_stats[bucket]["top5"] += a5 * n
                bucket_stats[bucket]["samples"] += 1
                total_loss += sample_loss * n
                total_valid += n
                if len(examples) < int(cfg["examples_per_eval"]):
                    pred = logits[i].argmax(dim=-1).detach().cpu().tolist()
                    examples.append({"id": sample["id"], "bucket": bucket, "text": sample["input_text"], "labels": labels[i].detach().cpu().tolist(), "pred": pred})
    metrics = {
        bucket: {
            "samples": s["samples"],
            "valid_spans": s["valid"],
            "loss": s["loss_sum"] / max(1, s["valid"]),
            "top1": s["top1"] / max(1, s["valid"]),
            "top5": s["top5"] / max(1, s["valid"]),
        }
        for bucket, s in sorted(bucket_stats.items())
    }
    out = {
        "split": split,
        "loss": total_loss / max(1, total_valid),
        "valid_spans": total_valid,
        "metrics_by_bucket": metrics,
    }
    split_dir = out_dir / f"eval_{split}"
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "metrics_by_bucket.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_jsonl(split_dir / "examples.jsonl", examples)
    adapter.train()
    classifier.train()
    return out


def env(model_name: str, cfg: dict) -> dict:
    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "torch_cuda_version": torch.version.cuda,
        }
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "gpu": gpu, "model_name": model_name, "config": cfg}


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
    llm = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, trust_remote_code=True).to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    checksum0 = checksum_first_tensors(llm)
    hidden_size = llm.get_input_embeddings().embedding_dim
    adapter_cfg = BoundaryByteAdapterConfig(
        byte_embed_dim=int(cfg["byte_embed_dim"]),
        hidden_dim=int(cfg["adapter_hidden_dim"]),
        llm_hidden_size=hidden_size,
        max_bytes=int(cfg["max_prompt_bytes"]),
        max_spans=int(cfg["max_spans"]),
        depth=int(cfg["adapter_depth"]),
        dropout=float(cfg["dropout"]),
    )
    adapter = BoundaryByteAdapter(adapter_cfg).to(device)
    classifier = nn.Linear(hidden_size, len(tokenizer)).to(device)
    train_rows = filter_rows(read_jsonl(cfg["train_manifest_path"]), list(cfg["train_buckets"]))
    val_rows = read_jsonl(cfg["validation_manifest_path"])
    loader = DataLoader(train_rows, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=lambda b: collate_recon(b, tokenizer, cfg))
    optimizer = AdamW(list(adapter.parameters()) + list(classifier.parameters()), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train_iter = iter(loader)
    first_loss = None
    last_row = None
    with open(out_dir / "train_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "top1", "top5", "valid_spans", "adapter_grad_abs_sum", "llm_grad_abs_sum", "gpu_mem_allocated_mb", "gpu_mem_reserved_mb", "elapsed_sec"])
        writer.writeheader()
        for step in range(1, int(cfg["steps"]) + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)
            latents, _ = adapter(batch["byte_ids"].to(device), batch["span_bounds"].to(device), batch["span_mask"].to(device))
            logits = classifier(latents)
            labels = batch["labels"].to(device)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            adapter_grad = grad_checksum(adapter.parameters())
            llm_grad = grad_checksum(llm.parameters())
            optimizer.step()
            top1, top5, valid_n = accuracy(logits.detach(), labels)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "top1": top1,
                "top5": top5,
                "valid_spans": valid_n,
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
    val_eval = evaluate(adapter, classifier, tokenizer, val_rows, cfg, device, out_dir, "validation")
    checksum1 = checksum_first_tensors(llm)
    summary = {
        "run_name": cfg["run_name"],
        "boundary_mode": cfg["boundary_mode"],
        "steps": int(cfg["steps"]),
        "train_rows": len(train_rows),
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "classifier_parameters": sum(p.numel() for p in classifier.parameters()),
        "llm_trainable_params": sum(p.numel() for p in llm.parameters() if p.requires_grad),
        "llm_checksum_delta_first8_tensors": checksum1 - checksum0,
        "initial_loss": first_loss,
        "final_loss": last_row["loss"],
        "loss_delta": first_loss - last_row["loss"],
        "final_top1": last_row["top1"],
        "final_top5": last_row["top5"],
        "validation": val_eval,
        "gpu_memory_allocated_peak_mib": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0,
        "gpu_memory_reserved_peak_mib": torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0,
        "total_elapsed_sec": time.perf_counter() - start,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env(cfg["model_name"], cfg), f, indent=2, ensure_ascii=False)
    ckpt = {"adapter_state_dict": adapter.state_dict(), "adapter_config": adapter_cfg.__dict__, "model_name": cfg["model_name"], "boundary_mode": cfg["boundary_mode"], "classifier_state_dict": classifier.state_dict(), "summary": summary}
    torch.save(ckpt, out_dir / "adapter.pt")
    torch.save(ckpt, out_dir / "final_adapter.pt")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
