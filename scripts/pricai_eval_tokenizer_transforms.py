from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import unicodedata
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


ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
REPEATED_ALNUM_RE = re.compile(r"([A-Za-z0-9])\1{2,}")
REPEATED_ALPHA_RE = re.compile(r"([A-Za-z])\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")
HORIZONTAL_SPACE_RE = re.compile(r"[\t\f\v \u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]+")
PUNCT_FOLD = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "…": "...",
    }
)


def transform_text(text: str, mode: str) -> str:
    """Apply deterministic tokenizer-side input cleanup without touching targets."""
    if mode == "raw":
        return text
    if mode in {
        "nfc",
        "nfc_cleanup",
        "whitespace",
        "punctfold",
        "casefold",
        "repeat_squeeze",
        "typo_cleanup",
    }:
        text = unicodedata.normalize("NFC", text)
    elif mode in {"nfkc", "nfkc_cleanup"}:
        text = unicodedata.normalize("NFKC", text)
    elif mode not in {"cleanup"}:
        raise ValueError(f"unknown transform mode: {mode}")

    if mode in {"whitespace", "typo_cleanup"}:
        text = HORIZONTAL_SPACE_RE.sub(" ", text)
        text = re.sub(r" *\n *", "\n", text).strip(" ")
    if mode in {"punctfold", "typo_cleanup"}:
        text = text.translate(PUNCT_FOLD)
    if mode in {"casefold", "typo_cleanup"}:
        text = text.casefold()
    if mode in {"repeat_squeeze", "typo_cleanup"}:
        text = REPEATED_ALPHA_RE.sub(lambda m: m.group(1) * 2, text)
    if mode in {"cleanup", "nfc_cleanup", "nfkc_cleanup"}:
        text = ZERO_WIDTH_RE.sub("", text)
        text = WHITESPACE_RE.sub(" ", text).strip()
        # Collapse long repeated alphanumeric runs without guessing a dictionary word.
        text = REPEATED_ALNUM_RE.sub(lambda m: m.group(1) * 2, text)
    return text


def transform_sample(sample: dict, mode: str) -> str:
    if mode == "oracle_clean_typo" and str(sample.get("bucket", "")).startswith("typo_noise"):
        return sample["target_text"]
    if mode == "oracle_clean_typo":
        return sample["input_text"]
    return transform_text(sample["input_text"], mode)


def collate(batch: list[dict], tokenizer, cfg: dict) -> dict:
    input_rows = []
    label_rows = []
    prompt_lengths = []
    target_lengths = []
    transformed_inputs = []
    mode = str(cfg["transform_mode"])
    for sample in batch:
        transformed_input = transform_sample(sample, mode)
        transformed_inputs.append(transformed_input)
        prompt_ids = tokenizer(
            transformed_input + str(cfg.get("prompt_suffix", "\n")),
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        target_ids = tokenizer(sample["target_text"], add_special_tokens=False, truncation=False)["input_ids"]
        prompt_ids = prompt_ids[: int(cfg["max_prompt_length"])]
        target_ids = target_ids[: int(cfg["max_target_length"])]
        input_rows.append(torch.tensor(prompt_ids + target_ids, dtype=torch.long))
        label_rows.append(torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long))
        prompt_lengths.append(len(prompt_ids))
        target_lengths.append(len(target_ids))
    input_ids = pad_sequence(input_rows, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(label_rows, batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    return {
        "samples": batch,
        "transformed_inputs": transformed_inputs,
        "encoded": {"input_ids": input_ids, "attention_mask": attention_mask},
        "labels": labels,
        "prompt_lengths": prompt_lengths,
        "target_lengths": target_lengths,
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
    parser.add_argument("--mode", default=None, help="Override transform_mode from config.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.mode:
        cfg["transform_mode"] = args.mode
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

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
        collate_fn=lambda b: collate(b, tokenizer, cfg),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sample_metrics = []
    total_tokens = 0
    changed = 0
    with torch.no_grad():
        for batch in loader:
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            labels = batch["labels"].to(device)
            outputs = model(**encoded)
            losses, token_counts = causal_lm_loss_per_sample(outputs.logits, labels)
            for i, (sample, transformed_input, loss, token_count) in enumerate(
                zip(batch["samples"], batch["transformed_inputs"], losses.cpu().tolist(), token_counts.cpu().tolist())
            ):
                total_tokens += int(token_count)
                input_changed = transformed_input != sample["input_text"]
                changed += int(input_changed)
                sample_metrics.append(
                    {
                        "id": sample["id"],
                        "clean_id": sample.get("clean_id"),
                        "bucket": sample["bucket"],
                        "source": sample["source"],
                        "perturbation_type": sample["perturbation_type"],
                        "noise_level": sample["noise_level"],
                        "transform_mode": cfg["transform_mode"],
                        "input_changed": input_changed,
                        "loss": float(loss),
                        "token_count": int(token_count),
                        "target_token_count": int(token_count),
                        "byte_count": len(transformed_input.encode("utf-8")),
                        "original_input_byte_len": len(sample["input_text"].encode("utf-8")),
                        "transformed_input_byte_len": len(transformed_input.encode("utf-8")),
                        "baseline_prompt_tokens": batch["prompt_lengths"][i],
                        "target_tokens": batch["target_lengths"][i],
                        "input_text": sample["input_text"],
                        "transformed_input_text": transformed_input,
                        "target_text": sample["target_text"],
                    }
                )

    elapsed = time.perf_counter() - start
    metrics_by_bucket = aggregate_by_bucket(sample_metrics)
    summary = {
        "run_name": f"tokenizer_{cfg['transform_mode']}",
        "model_name": cfg["model_name"],
        "manifest_path": cfg["manifest_path"],
        "transform_mode": cfg["transform_mode"],
        "num_samples": len(sample_metrics),
        "num_inputs_changed": changed,
        "fraction_inputs_changed": changed / max(1, len(sample_metrics)),
        "metrics_by_bucket": metrics_by_bucket,
        "tokens_per_sec": total_tokens / max(elapsed, 1e-9),
        "elapsed_sec": elapsed,
        "trainable_params": 0,
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
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
