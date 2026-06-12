from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "font.size": 8,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_vector(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")


def make_fragmentation() -> None:
    rows = read_csv(ROOT / "bucket_fragmentation.csv")
    order = [
        "clean_english",
        "code",
        "multilingual_ar",
        "multilingual_hi",
        "multilingual_ja",
        "multilingual_ko",
        "multilingual_latin",
        "multilingual_ru",
        "multilingual_zh",
        "tokenizer_stress",
        "typo_noise_heavy",
        "typo_noise_light",
        "typo_noise_medium",
        "unicode_stress",
    ]
    by_bucket = {r["bucket"]: r for r in rows}
    rows = [by_bucket[b] for b in order]

    labels = [r["bucket"] for r in rows]
    x = list(range(len(rows)))
    tokenizer_loss = [float(r["mean_tokenizer_loss"]) for r in rows]
    prefix_loss = [float(r["mean_phase5_prefix_p64_loss"]) for r in rows]
    kv_loss = [float(r["mean_phase6_kv_p32_all_loss"]) for r in rows]
    tokens_per_byte = [float(r["mean_tokens_per_byte"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.1, 3.0))
    width = 0.22
    colors = {
        "tok": "#4C78A8",
        "prefix": "#F58518",
        "kv": "#54A24B",
        "line": "#111111",
    }
    ax1.bar([i - width for i in x], tokenizer_loss, width=width, color=colors["tok"], label="tokenizer loss")
    ax1.bar(x, prefix_loss, width=width, color=colors["prefix"], label="soft prefix p64 loss")
    ax1.bar([i + width for i in x], kv_loss, width=width, color=colors["kv"], label="KV p32 all loss")
    ax1.set_ylabel("target-token loss")
    ax1.set_ylim(0, 11.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.tick_params(axis="x", labelsize=7.2)

    ax2 = ax1.twinx()
    ax2.plot(x, tokens_per_byte, color=colors["line"], marker="o", linewidth=1.3, markersize=3.2, label="tokens/byte")
    ax2.set_ylabel("input tokens per byte")
    ax2.set_ylim(0.2, 0.62)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=7, frameon=True, borderpad=0.35)
    fig.tight_layout(pad=0.4)
    save_vector(fig, ROOT / "bucket_loss_and_fragmentation_vector")
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    lines: list[str],
    face: str = "#FFFFFF",
    edge: str = "#111111",
    lw: float = 1.0,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.73, title, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.40, "\n".join(lines), ha="center", va="center", fontsize=8.4, linespacing=1.15)


def add_related_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    lines: list[str],
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.0,
            edgecolor="#111111",
            facecolor="#FFFFFF",
        )
    )
    ax.text(x + w / 2, y + h * 0.76, title, ha="center", va="center", fontsize=10.4, fontweight="bold")
    start = y + h * 0.54
    step = h * 0.15
    for i, line in enumerate(lines):
        ax.text(x + w / 2, start - i * step, line, ha="center", va="center", fontsize=8.9)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.0, color="#111111"))


def make_related_work() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Match the user-provided visual layout, but render every object as vector.
    add_related_box(ax, (0.02, 0.45), (0.30, 0.44), "Byte-capable models",
                    ["CANINE / ByT5", "MEGABYTE / BLT / MrT5", r"$\it{trained\ for\ raw\ units}$"])
    add_related_box(ax, (0.36, 0.45), (0.28, 0.44), "ByteBridge",
                    ["input-side byte adapter", r"$\it{bytes\ condition}$", r"$\it{a\ frozen\ LM}$"])
    add_related_box(ax, (0.68, 0.45), (0.30, 0.44), "Frozen-model adaptation",
                    ["prompt / prefix / LoRA", r"$\it{input\ usually}$", r"$\it{tokenized}$"])
    add_related_box(ax, (0.12, 0.10), (0.76, 0.21), "Tokenizer analysis",
                    ["motivates stress tests; no byte interface is trained"])

    arrow(ax, (0.322, 0.67), (0.358, 0.67))
    arrow(ax, (0.678, 0.67), (0.642, 0.67))

    fig.tight_layout(pad=0.2)
    save_vector(fig, ROOT / "bytebridge_related_work_vector")
    plt.close(fig)


def main() -> None:
    make_fragmentation()
    make_related_work()


if __name__ == "__main__":
    main()
