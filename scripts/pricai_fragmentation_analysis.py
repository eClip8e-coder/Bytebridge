from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bytebridge.data import read_jsonl


def load_sample_metrics(path: str | Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path)}


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(ranks(xs), ranks(ys))


def mean(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(1, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/phase2/splits/test.jsonl")
    parser.add_argument("--tokenizer-metrics", default="experiments/phase2/tokenizer_baseline/sample_metrics.jsonl")
    parser.add_argument("--run-metrics", nargs="+", required=True, help="Name=path pairs for ByteBridge runs.")
    parser.add_argument("--out-dir", default="experiments/pricai/fragmentation")
    args = parser.parse_args()

    manifest = {row["id"]: row for row in read_jsonl(args.manifest)}
    tokenizer_metrics = load_sample_metrics(args.tokenizer_metrics)
    runs = {}
    for item in args.run_metrics:
        if "=" not in item:
            raise ValueError(f"run metric must be Name=path, got {item}")
        name, path = item.split("=", 1)
        runs[name] = load_sample_metrics(path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id, sample in manifest.items():
        if sample_id not in tokenizer_metrics:
            continue
        input_bytes = float(sample.get("input_byte_len") or len(sample["input_text"].encode("utf-8")))
        input_tokens = float(sample.get("input_token_len") or tokenizer_metrics[sample_id].get("baseline_prompt_tokens", 0))
        target_tokens = float(sample.get("target_token_len") or tokenizer_metrics[sample_id].get("target_tokens", 0))
        base_loss = float(tokenizer_metrics[sample_id]["loss"])
        row = {
            "id": sample_id,
            "bucket": sample["bucket"],
            "input_byte_len": input_bytes,
            "input_token_len": input_tokens,
            "target_token_len": target_tokens,
            "tokens_per_byte": input_tokens / max(input_bytes, 1.0),
            "bytes_per_token": input_bytes / max(input_tokens, 1.0),
            "tokenizer_loss": base_loss,
        }
        for name, metrics in runs.items():
            if sample_id in metrics:
                loss = float(metrics[sample_id]["loss"])
                row[f"{name}_loss"] = loss
                row[f"{name}_gain_vs_tokenizer"] = (base_loss - loss) / max(base_loss, 1e-9)
                row[f"{name}_delta_vs_tokenizer"] = loss - base_loss
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(out_dir / "sample_fragmentation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["bucket"]].append(row)
    bucket_rows = []
    for bucket, bucket_items in sorted(grouped.items()):
        out = {
            "bucket": bucket,
            "count": len(bucket_items),
            "mean_input_byte_len": mean(bucket_items, "input_byte_len"),
            "mean_input_token_len": mean(bucket_items, "input_token_len"),
            "mean_tokens_per_byte": mean(bucket_items, "tokens_per_byte"),
            "mean_bytes_per_token": mean(bucket_items, "bytes_per_token"),
            "mean_tokenizer_loss": mean(bucket_items, "tokenizer_loss"),
        }
        for name in runs:
            if f"{name}_loss" in bucket_items[0]:
                out[f"mean_{name}_loss"] = mean(bucket_items, f"{name}_loss")
                out[f"mean_{name}_gain_vs_tokenizer"] = mean(bucket_items, f"{name}_gain_vs_tokenizer")
        bucket_rows.append(out)
    with open(out_dir / "bucket_fragmentation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bucket_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bucket_rows)

    correlations = {}
    predictors = ["input_byte_len", "input_token_len", "tokens_per_byte", "bytes_per_token"]
    outcomes = ["tokenizer_loss"]
    for name in runs:
        outcomes.extend([f"{name}_loss", f"{name}_gain_vs_tokenizer", f"{name}_delta_vs_tokenizer"])
    for pred in predictors:
        correlations[pred] = {}
        xs = [float(r[pred]) for r in rows]
        for out in outcomes:
            valid = [r for r in rows if out in r]
            if not valid:
                continue
            xs_valid = [float(r[pred]) for r in valid]
            ys = [float(r[out]) for r in valid]
            correlations[pred][out] = {"pearson": pearson(xs_valid, ys), "spearman": spearman(xs_valid, ys)}

    summary = {
        "manifest": args.manifest,
        "tokenizer_metrics": args.tokenizer_metrics,
        "runs": {k: len(v) for k, v in runs.items()},
        "num_samples": len(rows),
        "correlations": correlations,
    }
    with open(out_dir / "fragmentation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
