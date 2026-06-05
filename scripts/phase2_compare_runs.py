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
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="experiments/phase2/tables")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    base_metrics = baseline["metrics_by_bucket"]
    clean_base = base_metrics["clean_english"]["loss"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_summary in args.runs:
        summary = load_json(run_summary)
        run_name = summary["run_name"]
        run_metrics = summary["test_metrics_by_bucket"]
        clean_loss = run_metrics["clean_english"]["loss"]
        for bucket, metrics in sorted(run_metrics.items()):
            base_loss = base_metrics[bucket]["loss"]
            loss = metrics["loss"]
            rows.append(
                {
                    "run": run_name,
                    "bucket": bucket,
                    "baseline_loss": base_loss,
                    "bytebridge_loss": loss,
                    "loss_delta_bytebridge_minus_baseline": loss - base_loss,
                    "relative_robustness_gain_vs_baseline": (base_loss - loss) / base_loss,
                    "baseline_degradation_ratio": base_metrics[bucket]["degradation_ratio_vs_clean_bucket"],
                    "bytebridge_degradation_ratio": metrics["degradation_ratio_vs_clean_bucket"],
                    "clean_retention": clean_loss / clean_base,
                    "baseline_ppl": base_metrics[bucket]["ppl"],
                    "bytebridge_ppl": metrics["ppl"],
                    "count": metrics["count"],
                }
            )

    with open(out_dir / "phase2_pilot_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_run = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)
    summary_rows = []
    for run, run_rows in by_run.items():
        wins = [r for r in run_rows if r["relative_robustness_gain_vs_baseline"] > 0]
        noisy_wins = [
            r
            for r in run_rows
            if r["bucket"] != "clean_english" and r["relative_robustness_gain_vs_baseline"] > 0
        ]
        summary_rows.append(
            {
                "run": run,
                "clean_retention": run_rows[0]["clean_retention"],
                "num_bucket_wins": len(wins),
                "num_nonclean_bucket_wins": len(noisy_wins),
                "winning_buckets": [r["bucket"] for r in wins],
                "phase2_clean_retention_pass_1p2": run_rows[0]["clean_retention"] <= 1.2,
            }
        )
    with open(out_dir / "phase2_pilot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary_rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
