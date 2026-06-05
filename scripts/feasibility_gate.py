from __future__ import annotations

import argparse
import csv
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
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from bytebridge.adapters import ByteAdapter, ByteAdapterConfig
from bytebridge.data import build_toy_corpus, encode_bytes_batch
from bytebridge.utils import seed_everything


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def try_load_model(model_name: str, fallback_model_name: str | None, device: torch.device):
    errors: list[str] = []
    for name in [model_name, fallback_model_name]:
        if not name:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            model.to(device)
            return name, tokenizer, model, errors
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load any model. Errors:\n" + "\n".join(errors))


def build_labels(tokenizer, texts: list[str], seq_len: int, device: torch.device) -> torch.Tensor:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    labels = encoded["input_ids"]
    labels[encoded["attention_mask"].eq(0)] = -100
    return labels.to(device)


def grad_checksum(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().abs().sum().cpu())
    return total


def collect_env(model_name: str, actual_model_name: str, config: dict, load_errors: list[str]) -> dict:
    gpu = {}
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        gpu = {
            "name": torch.cuda.get_device_name(idx),
            "total_memory_bytes": torch.cuda.get_device_properties(idx).total_memory,
            "torch_cuda_version": torch.version.cuda,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(idx),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(idx),
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "requested_model_name": model_name,
        "actual_model_name": actual_model_name,
        "model_load_errors": load_errors,
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feasibility_qwen05.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    actual_model_name, tokenizer, llm, load_errors = try_load_model(
        config["model_name"],
        config.get("fallback_model_name"),
        device,
    )
    llm.eval()
    for param in llm.parameters():
        param.requires_grad_(False)

    hidden_size = llm.get_input_embeddings().embedding_dim
    llm_input_dtype = llm.get_input_embeddings().weight.dtype
    adapter_config = ByteAdapterConfig(
        byte_embed_dim=int(config["byte_embed_dim"]),
        hidden_dim=int(config["adapter_hidden_dim"]),
        llm_hidden_size=hidden_size,
        patch_size=int(config["patch_size"]),
        max_bytes=int(config["max_bytes"]),
        depth=int(config["adapter_depth"]),
        dropout=float(config["dropout"]),
    )
    adapter = ByteAdapter(adapter_config).to(device)
    adapter.train()

    texts = build_toy_corpus()
    seq_len = adapter.num_patches
    byte_ids = encode_bytes_batch(texts, adapter_config.max_bytes).to(device)
    labels = build_labels(tokenizer, texts, seq_len, device)

    optimizer = AdamW(
        adapter.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    metrics_path = out_dir / "metrics.csv"
    log_rows: list[dict] = []
    start_time = time.perf_counter()
    initial_llm_checksum = sum(float(p.detach().float().sum().cpu()) for p in list(llm.parameters())[:4])

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "adapter_grad_abs_sum",
                "llm_grad_abs_sum",
                "gpu_mem_allocated_mb",
                "gpu_mem_reserved_mb",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        for step in range(1, int(config["steps"]) + 1):
            indices = torch.randint(0, len(texts), (int(config["batch_size"]),), device=device)
            batch_bytes = byte_ids.index_select(0, indices)
            batch_labels = labels.index_select(0, indices)
            inputs_embeds, patch_mask = adapter(batch_bytes)
            inputs_embeds = inputs_embeds.to(dtype=llm_input_dtype)

            outputs = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=patch_mask.long(),
                labels=batch_labels,
            )
            loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            adapter_grad = grad_checksum(adapter.parameters())
            llm_grad = grad_checksum(llm.parameters())
            optimizer.step()

            if device.type == "cuda":
                mem_alloc = torch.cuda.max_memory_allocated() / (1024**2)
                mem_reserved = torch.cuda.max_memory_reserved() / (1024**2)
            else:
                mem_alloc = 0.0
                mem_reserved = 0.0
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "adapter_grad_abs_sum": adapter_grad,
                "llm_grad_abs_sum": llm_grad,
                "gpu_mem_allocated_mb": mem_alloc,
                "gpu_mem_reserved_mb": mem_reserved,
                "elapsed_sec": time.perf_counter() - start_time,
            }
            writer.writerow(row)
            f.flush()
            log_rows.append(row)
            if step == 1 or step % int(config["log_every"]) == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)

    final_llm_checksum = sum(float(p.detach().float().sum().cpu()) for p in list(llm.parameters())[:4])
    env = collect_env(config["model_name"], actual_model_name, config, load_errors)
    env.update(
        {
            "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
            "trainable_llm_parameters": sum(p.numel() for p in llm.parameters() if p.requires_grad),
            "num_patches": adapter.num_patches,
            "compression_ratio_bytes_per_latent": adapter_config.patch_size,
            "batch_size": int(config["batch_size"]),
            "sequence_length_latents": adapter.num_patches,
            "total_train_time_sec": time.perf_counter() - start_time,
            "initial_loss": log_rows[0]["loss"],
            "final_loss": log_rows[-1]["loss"],
            "loss_delta": log_rows[0]["loss"] - log_rows[-1]["loss"],
            "llm_checksum_delta_first4_tensors": final_llm_checksum - initial_llm_checksum,
        }
    )
    with open(out_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "actual_model_name": actual_model_name,
                "initial_loss": log_rows[0]["loss"],
                "final_loss": log_rows[-1]["loss"],
                "loss_delta": log_rows[0]["loss"] - log_rows[-1]["loss"],
                "adapter_grad_seen": max(r["adapter_grad_abs_sum"] for r in log_rows) > 0,
                "llm_grad_seen": max(r["llm_grad_abs_sum"] for r in log_rows) > 0,
                "llm_checksum_delta_first4_tensors": final_llm_checksum - initial_llm_checksum,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    if bool(config.get("save_checkpoint", True)):
        torch.save(
            {
                "adapter_state_dict": adapter.state_dict(),
                "adapter_config": adapter_config.__dict__,
                "model_name": actual_model_name,
            },
            out_dir / "adapter.pt",
        )


if __name__ == "__main__":
    main()
