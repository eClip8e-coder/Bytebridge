from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragmentation-csv", default="experiments/pricai/fragmentation/bucket_fragmentation.csv")
    parser.add_argument("--transform-csv", default="experiments/pricai/tables/tokenizer_transform_comparison.csv")
    parser.add_argument("--out-dir", default="experiments/pricai/plots")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frag_rows = read_csv(args.fragmentation_csv)
    buckets = [r["bucket"] for r in frag_rows]
    x = list(range(len(buckets)))
    tokenizer_loss = [float(r["mean_tokenizer_loss"]) for r in frag_rows]
    tokens_per_byte = [float(r["mean_tokens_per_byte"]) for r in frag_rows]
    prefix_loss = [float(r["mean_phase5_prefix_p64_loss"]) for r in frag_rows]
    kv_loss = [float(r["mean_phase6_kv_p32_all_loss"]) for r in frag_rows]

    fig, ax1 = plt.subplots(figsize=(11, 4.2))
    ax1.bar([i - 0.2 for i in x], tokenizer_loss, width=0.2, label="tokenizer loss")
    ax1.bar(x, prefix_loss, width=0.2, label="soft prefix p64 loss")
    ax1.bar([i + 0.2 for i in x], kv_loss, width=0.2, label="KV p32 all loss")
    ax1.set_ylabel("target-token loss")
    ax1.set_xticks(x)
    ax1.set_xticklabels(buckets, rotation=45, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, tokens_per_byte, color="black", marker="o", linewidth=1.5, label="tokens/byte")
    ax2.set_ylabel("input tokens per byte")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "bucket_loss_and_fragmentation.png", dpi=180)
    plt.close(fig)

    transform_rows = read_csv(args.transform_csv)
    modes = ["raw", "whitespace", "repeat_squeeze", "casefold", "typo_cleanup", "oracle_clean_typo"]
    rows = [r for r in transform_rows if r["mode"] in modes]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.22
    positions = list(range(len(rows)))
    for offset, bucket, label in [
        (-width, "typo_noise_light", "light"),
        (0.0, "typo_noise_medium", "medium"),
        (width, "typo_noise_heavy", "heavy"),
    ]:
        ax.bar([p + offset for p in positions], [float(r[bucket]) for r in rows], width=width, label=label)
    ax.set_xticks(positions)
    ax.set_xticklabels([r["mode"] for r in rows], rotation=25, ha="right")
    ax.set_ylabel("target-token loss")
    ax.set_title("Tokenizer preprocessing controls on typo buckets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "preprocessing_typo_controls.png", dpi=180)
    plt.close(fig)

    summary = {
        "plots": [
            str(out_dir / "bucket_loss_and_fragmentation.png"),
            str(out_dir / "preprocessing_typo_controls.png"),
        ]
    }
    with open(out_dir / "plot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
