from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from bytebridge.data import read_jsonl, write_jsonl
from bytebridge.eval import aggregate_by_bucket, causal_lm_loss_per_sample


def collate(batch: list[dict], tokenizer, max_target_length: int) -> dict:
    start_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    input_rows = []
    label_rows = []
    target_lengths = []
    target_truncated = []
    for sample in batch:
        target = tokenizer(sample["target_text"], add_special_tokens=False, truncation=False)["input_ids"]
        raw_len = len(target)
        target = target[:max_target_length]
        input_rows.append(torch.tensor([start_id] + target, dtype=torch.long))
        label_rows.append(torch.tensor([-100] + target, dtype=torch.long))
        target_lengths.append(len(target))
        target_truncated.append(raw_len > len(target))
    input_ids = pad_sequence(input_rows, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(label_rows, batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    return {
        "samples": batch,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_lengths": target_lengths,
        "target_truncated": target_truncated,
    }


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = read_jsonl(cfg["manifest_path"])
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer, int(cfg["max_target_length"])),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sample_metrics = []
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            losses, token_counts = causal_lm_loss_per_sample(outputs.logits, labels)
            for i, (sample, loss, token_count) in enumerate(zip(batch["samples"], losses.cpu().tolist(), token_counts.cpu().tolist())):
                total_tokens += int(token_count)
                sample_metrics.append(
                    {
                        "id": sample["id"],
                        "clean_id": sample.get("clean_id"),
                        "bucket": sample["bucket"],
                        "loss": float(loss),
                        "token_count": int(token_count),
                        "target_token_count": int(token_count),
                        "target_tokens": batch["target_lengths"][i],
                        "target_truncated": batch["target_truncated"][i],
                        "input_text": sample["input_text"],
                        "target_text": sample["target_text"],
                    }
                )

    elapsed = time.perf_counter() - start
    metrics_by_bucket = aggregate_by_bucket(sample_metrics)
    summary = {
        "model_name": cfg["model_name"],
        "manifest_path": cfg["manifest_path"],
        "num_samples": len(sample_metrics),
        "metrics_by_bucket": metrics_by_bucket,
        "tokens_per_sec": total_tokens / max(elapsed, 1e-9),
        "elapsed_sec": elapsed,
        "llm_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "gpu_memory_allocated_peak_mib": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0,
        "gpu_memory_reserved_peak_mib": torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0,
    }
    write_jsonl(out_dir / "sample_metrics.jsonl", sample_metrics)
    with open(out_dir / "metrics_by_bucket.json", "w", encoding="utf-8") as f:
        json.dump(metrics_by_bucket, f, indent=2, ensure_ascii=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env(cfg["model_name"], cfg), f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
