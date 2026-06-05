from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "qwen05_kv_p32_all": ROOT / "experiments/phase6/kv_prefix_p32_lall",
    "tinyllama_kv_p32_all_s71": ROOT / "experiments/pricai/tinyllama/kv_prefix_p32_lall",
    "tinyllama_kv_p32_all_s72": ROOT / "experiments/pricai/tinyllama/kv_prefix_p32_lall_seed72",
    "tinyllama_kv_p32_all_s73": ROOT / "experiments/pricai/tinyllama/kv_prefix_p32_lall_seed73",
    "tinyllama_constant_kv": ROOT / "experiments/pricai/tinyllama/constant_kv_prefix_p32_lall",
}


def read_train_metrics(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validation_clean_losses(run_dir: Path) -> dict[int, float]:
    out = {}
    for path in run_dir.glob("eval_validation_step*/metrics_by_bucket.json"):
        step = int(path.parent.name.replace("eval_validation_step", ""))
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        out[step] = metrics["clean_english"]["loss"]
    return out


def main() -> None:
    out_dir = ROOT / "experiments/pricai/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    curve_rows = []
    for name, run_dir in RUNS.items():
        metrics = read_train_metrics(run_dir / "train_metrics.csv")
        vals = validation_clean_losses(run_dir)
        with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
            run_summary = json.load(f)
        first = metrics[0]
        last = metrics[-1]
        selected_steps = {1, 100, 500, 1000, 2000}
        for row in metrics:
            step = int(row["step"])
            if step in selected_steps:
                curve_rows.append(
                    {
                        "run": name,
                        "step": step,
                        "train_loss": float(row["loss"]),
                        "validation_clean_loss": vals.get(step),
                        "adapter_grad_abs_sum": float(row["adapter_grad_abs_sum"]),
                        "llm_grad_abs_sum": float(row["llm_grad_abs_sum"]),
                    }
                )
        summary[name] = {
            "first_train_loss": float(first["loss"]),
            "last_train_loss": float(last["loss"]),
            "train_loss_drop": float(first["loss"]) - float(last["loss"]),
            "best_validation_clean_loss": run_summary.get("best_validation_clean_loss"),
            "test_clean_loss": run_summary["test_metrics_by_bucket"]["clean_english"]["loss"],
            "llm_trainable_params": run_summary.get("llm_trainable_params", 0),
            "checksum_delta": run_summary.get("llm_checksum_delta_first8_tensors", run_summary.get("llm_checksum_delta_first4_tensors", 0.0)),
        }
    with open(out_dir / "kv_training_curve_points.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)
    with open(out_dir / "kv_training_curve_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
