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
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from bytebridge.adapters import BytePrefixAdapter, BytePrefixAdapterConfig
from bytebridge.data import encode_bytes_batch, read_jsonl, write_jsonl
from bytebridge.eval import aggregate_by_bucket, causal_lm_loss_per_sample
from bytebridge.utils import seed_everything
from scripts.phase2_train_bytebridge import checksum_first_tensors, filter_rows, grad_checksum


def collate_prefix(batch, tokenizer, cfg):
    prompt_texts = [sample["input_text"] + str(cfg.get("prompt_suffix", "\n")) for sample in batch]
    byte_ids = encode_bytes_batch(prompt_texts, int(cfg["max_prompt_bytes"]))
    target_rows = []
    target_lengths = []
    target_truncated = []
    for sample in batch:
        ids = tokenizer(sample["target_text"], add_special_tokens=False, truncation=False)["input_ids"]
        raw_len = len(ids)
        ids = ids[: int(cfg["max_target_length"])]
        target_rows.append(torch.tensor(ids, dtype=torch.long))
        target_lengths.append(len(ids))
        target_truncated.append(raw_len > len(ids))
    target_ids = pad_sequence(target_rows, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_mask = target_ids.ne(tokenizer.pad_token_id)
    return {
        "samples": batch,
        "byte_ids": byte_ids,
        "target_ids": target_ids,
        "target_mask": target_mask,
        "target_lengths": target_lengths,
        "target_truncated": target_truncated,
    }


def forward_batch(batch, adapter, llm, tokenizer, llm_dtype, device):
    prefix_embeds, prefix_mask = adapter(batch["byte_ids"].to(device))
    prefix_embeds = prefix_embeds.to(dtype=llm_dtype)
    target_ids = batch["target_ids"].to(device)
    target_mask = batch["target_mask"].to(device)
    safe_target_ids = target_ids.masked_fill(~target_mask, tokenizer.pad_token_id)
    target_embeds = llm.get_input_embeddings()(safe_target_ids)
    inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)
    attention_mask = torch.cat([prefix_mask.to(device), target_mask], dim=1).long()
    labels = torch.cat(
        [
            torch.full(prefix_mask.shape, -100, dtype=torch.long, device=device),
            target_ids.masked_fill(~target_mask, -100),
        ],
        dim=1,
    )
    return llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels), labels, prefix_mask


def evaluate(adapter, llm, tokenizer, rows, cfg, device, out_dir, split):
    adapter.eval()
    loader = DataLoader(rows, batch_size=int(cfg["eval_batch_size"]), shuffle=False, collate_fn=lambda b: collate_prefix(b, tokenizer, cfg))
    sample_metrics = []
    total_tokens = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            outputs, labels, prefix_mask = forward_batch(batch, adapter, llm, tokenizer, llm.get_input_embeddings().weight.dtype, device)
            losses, token_counts = causal_lm_loss_per_sample(outputs.logits, labels)
            prefix_latents = prefix_mask.sum(dim=1).cpu().tolist()
            for i, (sample, loss, token_count) in enumerate(zip(batch["samples"], losses.cpu().tolist(), token_counts.cpu().tolist())):
                total_tokens += int(token_count)
                sample_metrics.append(
                    {
                        "id": sample["id"],
                        "clean_id": sample.get("clean_id"),
                        "bucket": sample["bucket"],
                        "source": sample["source"],
                        "perturbation_type": sample["perturbation_type"],
                        "noise_level": sample["noise_level"],
                        "loss": float(loss),
                        "token_count": int(token_count),
                        "target_token_count": int(token_count),
                        "byte_count": len(sample["input_text"].encode("utf-8")),
                        "prefix_latents": int(prefix_latents[i]),
                        "target_tokens": batch["target_lengths"][i],
                        "target_truncated": batch["target_truncated"][i],
                        "input_text": sample["input_text"],
                        "target_text": sample["target_text"],
                    }
                )
    split_dir = out_dir / f"eval_{split}"
    split_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(split_dir / "sample_metrics.jsonl", sample_metrics)
    metrics = aggregate_by_bucket(sample_metrics)
    with open(split_dir / "metrics_by_bucket.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    adapter.train()
    return {"split": split, "num_samples": len(sample_metrics), "metrics_by_bucket": metrics, "tokens_per_sec": total_tokens / max(time.perf_counter() - start, 1e-9)}


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


def main():
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
    adapter_cfg = BytePrefixAdapterConfig(
        byte_embed_dim=int(cfg["byte_embed_dim"]),
        hidden_dim=int(cfg["adapter_hidden_dim"]),
        llm_hidden_size=llm.get_input_embeddings().embedding_dim,
        max_bytes=int(cfg["max_prompt_bytes"]),
        prefix_length=int(cfg["prefix_length"]),
        depth=int(cfg["adapter_depth"]),
        dropout=float(cfg["dropout"]),
        pooling_mode=str(cfg["pooling_mode"]),
    )
    adapter = BytePrefixAdapter(adapter_cfg).to(device)
    adapter.train()
    train_rows = filter_rows(read_jsonl(cfg["train_manifest_path"]), list(cfg["train_buckets"]))
    val_rows = read_jsonl(cfg["validation_manifest_path"])
    test_rows = read_jsonl(cfg["test_manifest_path"])
    loader = DataLoader(train_rows, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=lambda b: collate_prefix(b, tokenizer, cfg))
    optimizer = AdamW(adapter.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train_iter = iter(loader)
    best_val_loss = float("inf")
    with open(out_dir / "train_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "adapter_grad_abs_sum", "llm_grad_abs_sum", "gpu_mem_allocated_mb", "gpu_mem_reserved_mb", "elapsed_sec"])
        writer.writeheader()
        for step in range(1, int(cfg["steps"]) + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)
            outputs, _, _ = forward_batch(batch, adapter, llm, tokenizer, llm.get_input_embeddings().weight.dtype, device)
            loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            adapter_grad = grad_checksum(adapter.parameters())
            llm_grad = grad_checksum(llm.parameters())
            optimizer.step()
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "adapter_grad_abs_sum": adapter_grad,
                "llm_grad_abs_sum": llm_grad,
                "gpu_mem_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0,
                "gpu_mem_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0,
                "elapsed_sec": time.perf_counter() - start,
            }
            writer.writerow(row)
            if step == 1 or step % int(cfg["log_every"]) == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if step % int(cfg["eval_every"]) == 0 or step == int(cfg["steps"]):
                val_eval = evaluate(adapter, llm, tokenizer, val_rows, cfg, device, out_dir, f"validation_step{step}")
                val_clean = val_eval["metrics_by_bucket"].get("clean_english", {}).get("loss", float("inf"))
                if val_clean < best_val_loss:
                    best_val_loss = val_clean
                    torch.save({"adapter_state_dict": adapter.state_dict(), "adapter_config": adapter_cfg.__dict__, "model_name": cfg["model_name"], "step": step, "validation_clean_loss": val_clean}, out_dir / "best_adapter.pt")
    final_eval = evaluate(adapter, llm, tokenizer, test_rows, cfg, device, out_dir, "test_final")
    summary = {
        "run_name": cfg["run_name"],
        "model_name": cfg["model_name"],
        "prefix_length": int(cfg["prefix_length"]),
        "pooling_mode": cfg["pooling_mode"],
        "steps": int(cfg["steps"]),
        "train_rows": len(train_rows),
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "llm_trainable_params": sum(p.numel() for p in llm.parameters() if p.requires_grad),
        "llm_checksum_delta_first8_tensors": checksum_first_tensors(llm) - checksum0,
        "best_validation_clean_loss": best_val_loss,
        "test_metrics_by_bucket": final_eval["metrics_by_bucket"],
        "test_tokens_per_sec": final_eval["tokens_per_sec"],
        "gpu_memory_allocated_peak_mib": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0,
        "gpu_memory_reserved_peak_mib": torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0,
        "total_elapsed_sec": time.perf_counter() - start,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env(cfg["model_name"], cfg), f, indent=2, ensure_ascii=False)
    torch.save({"adapter_state_dict": adapter.state_dict(), "adapter_config": adapter_cfg.__dict__, "model_name": cfg["model_name"], "step": int(cfg["steps"])}, out_dir / "adapter.pt")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
