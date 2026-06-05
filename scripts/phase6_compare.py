from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="experiments/phase2/tokenizer_baseline/summary.json")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="experiments/phase6/tables")
    args = parser.parse_args()
    baseline = load_json(args.baseline)
    base_metrics = baseline["metrics_by_bucket"]
    clean_base = base_metrics["clean_english"]["loss"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    for path in args.runs:
        data = load_json(path)
        run = data.get("run_name", Path(path).parent.name)
        metrics = data["test_metrics_by_bucket"]
        clean_loss = metrics["clean_english"]["loss"]
        wins = []
        for bucket, m in sorted(metrics.items()):
            base_loss = base_metrics[bucket]["loss"]
            gain = (base_loss - m["loss"]) / base_loss
            if gain > 0:
                wins.append(bucket)
            rows.append({"run": run, "bucket": bucket, "baseline_loss": base_loss, "run_loss": m["loss"], "loss_delta_run_minus_baseline": m["loss"] - base_loss, "relative_gain_vs_baseline": gain, "clean_retention": clean_loss / clean_base, "count": m["count"]})
        summary.append({"run": run, "path": path, "clean_loss": clean_loss, "clean_retention": clean_loss / clean_base, "winning_buckets": wins, "num_winning_buckets": len(wins), "llm_trainable_params": data.get("llm_trainable_params"), "checksum_delta": data.get("llm_checksum_delta_first8_tensors"), "prefix_length": data.get("prefix_length"), "target_layers": data.get("target_layers"), "num_layers_injected": data.get("num_layers_injected"), "test_tokens_per_sec": data.get("test_tokens_per_sec"), "gpu_memory_allocated_peak_mib": data.get("gpu_memory_allocated_peak_mib"), "gpu_memory_reserved_peak_mib": data.get("gpu_memory_reserved_peak_mib")})
    with open(out_dir / "phase6_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "phase6_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
