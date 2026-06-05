from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from bytebridge.data import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen05": "Qwen/Qwen2.5-0.5B",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx and deny else float("nan")


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank(xs), rank(ys))


def load_losses(path: Path) -> dict[str, float]:
    losses = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                losses[row["id"]] = float(row["loss"])
    return losses


def main() -> None:
    rows = read_jsonl("data/phase2/splits/test.jsonl")
    out_dir = ROOT / "experiments/pricai/tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizers = {}
    for key, name in MODELS.items():
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tokenizers[key] = tok

    qwen_losses = load_losses(ROOT / "experiments/phase2/tokenizer_baseline/sample_metrics.jsonl")
    tiny_losses = load_losses(ROOT / "experiments/pricai/tinyllama/tokenizer_baseline/sample_metrics.jsonl")

    sample_rows = []
    for sample in rows:
        text = sample["input_text"]
        byte_len = max(len(text.encode("utf-8")), 1)
        char_len = max(len(text), 1)
        item = {
            "id": sample["id"],
            "bucket": sample["bucket"],
            "byte_len": byte_len,
            "char_len": char_len,
            "qwen_tokenizer_loss": qwen_losses[sample["id"]],
            "tinyllama_tokenizer_loss": tiny_losses[sample["id"]],
        }
        for key, tok in tokenizers.items():
            ids = tok(text + "\n", add_special_tokens=False, truncation=False)["input_ids"]
            tok_len = max(len(ids), 1)
            item[f"{key}_input_tokens"] = tok_len
            item[f"{key}_tokens_per_byte"] = tok_len / byte_len
            item[f"{key}_bytes_per_token"] = byte_len / tok_len
            item[f"{key}_chars_per_token"] = char_len / tok_len
        item["tiny_minus_qwen_tokens"] = item["tinyllama_input_tokens"] - item["qwen05_input_tokens"]
        sample_rows.append(item)

    bucket_rows = []
    for bucket in sorted({r["bucket"] for r in sample_rows}):
        b = [r for r in sample_rows if r["bucket"] == bucket]
        row = {"bucket": bucket, "count": len(b)}
        for key in MODELS:
            for field in ["input_tokens", "tokens_per_byte", "bytes_per_token", "chars_per_token"]:
                values = [r[f"{key}_{field}"] for r in b]
                row[f"{key}_{field}_mean"] = statistics.mean(values)
        row["tiny_minus_qwen_tokens_mean"] = statistics.mean(r["tiny_minus_qwen_tokens"] for r in b)
        row["qwen_loss_mean"] = statistics.mean(r["qwen_tokenizer_loss"] for r in b)
        row["tiny_loss_mean"] = statistics.mean(r["tinyllama_tokenizer_loss"] for r in b)
        bucket_rows.append(row)

    corr_rows = []
    for key, loss_field in [("qwen05", "qwen_tokenizer_loss"), ("tinyllama", "tinyllama_tokenizer_loss")]:
        for pred in ["input_tokens", "tokens_per_byte", "bytes_per_token", "chars_per_token"]:
            xs = [r[f"{key}_{pred}"] for r in sample_rows]
            ys = [r[loss_field] for r in sample_rows]
            corr_rows.append(
                {
                    "model": key,
                    "predictor": pred,
                    "loss": loss_field,
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )

    for filename, data in [
        ("tokenizer_fertility_samples.csv", sample_rows),
        ("tokenizer_fertility_by_bucket.csv", bucket_rows),
        ("tokenizer_fertility_correlations.csv", corr_rows),
    ]:
        with open(out_dir / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    summary = {
        "models": MODELS,
        "num_samples": len(sample_rows),
        "bucket_rows": bucket_rows,
        "correlations": corr_rows,
    }
    with open(out_dir / "tokenizer_fertility_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
