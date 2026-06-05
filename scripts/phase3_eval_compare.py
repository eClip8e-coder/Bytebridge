from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="experiments/phase2/tokenizer_baseline/summary.json")
    parser.add_argument("--phase2-best", default="experiments/phase2/bb_noise_l32_seed1_2k/summary.json")
    parser.add_argument("--phase3-runs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="experiments/phase3/tables")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    base_metrics = baseline["metrics_by_bucket"]
    clean_base = base_metrics["clean_english"]["loss"]
    run_paths = [args.phase2_best] + args.phase3_runs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = []
    for path in run_paths:
        data = load_json(path)
        run_name = data.get("run_name", Path(path).parent.name)
        metrics = data["test_metrics_by_bucket"]
        clean_loss = metrics["clean_english"]["loss"]
        wins = []
        for bucket, bucket_metrics in sorted(metrics.items()):
            base_loss = base_metrics[bucket]["loss"]
            loss = bucket_metrics["loss"]
            gain = (base_loss - loss) / base_loss
            if gain > 0:
                wins.append(bucket)
            rows.append(
                {
                    "run": run_name,
                    "bucket": bucket,
                    "baseline_loss": base_loss,
                    "bytebridge_loss": loss,
                    "loss_delta_bytebridge_minus_baseline": loss - base_loss,
                    "relative_robustness_gain_vs_baseline": gain,
                    "baseline_degradation_ratio": base_metrics[bucket]["degradation_ratio_vs_clean_bucket"],
                    "bytebridge_degradation_ratio": bucket_metrics["degradation_ratio_vs_clean_bucket"],
                    "clean_retention": clean_loss / clean_base,
                    "count": bucket_metrics["count"],
                }
            )
        summary.append(
            {
                "run": run_name,
                "path": path,
                "clean_loss": clean_loss,
                "clean_retention": clean_loss / clean_base,
                "winning_buckets": wins,
                "num_winning_buckets": len(wins),
                "clean_retention_le_1p5": clean_loss / clean_base <= 1.5,
                "clean_retention_le_1p2": clean_loss / clean_base <= 1.2,
                "llm_checksum_delta_first8_tensors": data.get("llm_checksum_delta_first8_tensors"),
                "llm_trainable_params": data.get("llm_trainable_params"),
            }
        )

    with open(out_dir / "phase3_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "phase3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
