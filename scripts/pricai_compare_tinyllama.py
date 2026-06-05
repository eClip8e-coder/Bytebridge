from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RUNS = [
    ("target_only", ROOT / "experiments/pricai/tinyllama/target_only_baseline/summary.json", None),
    ("tokenizer", ROOT / "experiments/pricai/tinyllama/tokenizer_baseline/summary.json", None),
    ("prefix_p64", ROOT / "experiments/pricai/tinyllama/prefix_fixed_p64/summary.json", "tokenizer"),
    ("kv_p32_all_s71", ROOT / "experiments/pricai/tinyllama/kv_prefix_p32_lall/summary.json", "tokenizer"),
    ("kv_p32_all_s72", ROOT / "experiments/pricai/tinyllama/kv_prefix_p32_lall_seed72/summary.json", "tokenizer"),
    ("kv_p32_all_s73", ROOT / "experiments/pricai/tinyllama/kv_prefix_p32_lall_seed73/summary.json", "tokenizer"),
]

BUCKETS = [
    "clean_english",
    "typo_noise_light",
    "typo_noise_medium",
    "typo_noise_heavy",
    "unicode_stress",
    "code",
    "tokenizer_stress",
    "multilingual_zh",
    "multilingual_ar",
]


def metrics(summary: dict) -> dict:
    return summary["metrics_by_bucket"] if "metrics_by_bucket" in summary else summary["test_metrics_by_bucket"]


def main() -> None:
    out_dir = ROOT / "experiments/pricai/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, path, _ in RUNS:
        with open(path, "r", encoding="utf-8") as f:
            summaries[name] = json.load(f)

    tokenizer_clean = metrics(summaries["tokenizer"])["clean_english"]["loss"]
    rows = []
    for name, _, compare_name in RUNS:
        summary = summaries[name]
        run_metrics = metrics(summary)
        clean_loss = run_metrics["clean_english"]["loss"]
        wins = []
        if compare_name:
            base_metrics = metrics(summaries[compare_name])
            for bucket in BUCKETS:
                if run_metrics[bucket]["loss"] < base_metrics[bucket]["loss"]:
                    wins.append(bucket)
        for bucket in BUCKETS:
            baseline_loss = metrics(summaries["tokenizer"])[bucket]["loss"]
            run_loss = run_metrics[bucket]["loss"]
            rows.append(
                {
                    "run": name,
                    "bucket": bucket,
                    "tokenizer_loss": baseline_loss,
                    "run_loss": run_loss,
                    "delta_vs_tokenizer": run_loss - baseline_loss,
                    "relative_gain_vs_tokenizer": (baseline_loss - run_loss) / baseline_loss,
                    "clean_retention_vs_tokenizer": clean_loss / tokenizer_clean,
                    "llm_trainable_params": summary.get("llm_trainable_params", 0),
                    "checksum_delta": summary.get("llm_checksum_delta_first8_tensors", 0.0),
                }
            )
        summary["clean_retention_vs_tokenizer"] = clean_loss / tokenizer_clean
        summary["wins_vs_tokenizer_selected"] = wins

    with open(out_dir / "tinyllama_second_family_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    kv_seed_names = ["kv_p32_all_s71", "kv_p32_all_s72", "kv_p32_all_s73"]
    kv_seed_summary = {
        "clean_retention_values": [summaries[name]["clean_retention_vs_tokenizer"] for name in kv_seed_names],
        "bucket_loss_values": {
            bucket: [metrics(summaries[name])[bucket]["loss"] for name in kv_seed_names]
            for bucket in BUCKETS
        },
    }
    kv_seed_summary["clean_retention_mean"] = sum(kv_seed_summary["clean_retention_values"]) / len(kv_seed_names)
    kv_seed_summary["bucket_loss_mean"] = {
        bucket: sum(values) / len(values)
        for bucket, values in kv_seed_summary["bucket_loss_values"].items()
    }
    tokenizer_metrics = metrics(summaries["tokenizer"])
    kv_seed_summary["mean_wins_vs_tokenizer"] = [
        bucket for bucket, loss in kv_seed_summary["bucket_loss_mean"].items()
        if loss < tokenizer_metrics[bucket]["loss"]
    ]
    summaries["kv_p32_all_3seed_mean"] = kv_seed_summary

    with open(out_dir / "tinyllama_second_family_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "prefix_p64": {
            "clean_retention": summaries["prefix_p64"]["clean_retention_vs_tokenizer"],
            "wins": summaries["prefix_p64"]["wins_vs_tokenizer_selected"],
        },
        "kv_p32_all_3seed_mean": {
            "clean_retention": kv_seed_summary["clean_retention_mean"],
            "wins": kv_seed_summary["mean_wins_vs_tokenizer"],
            "clean_retention_values": kv_seed_summary["clean_retention_values"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
