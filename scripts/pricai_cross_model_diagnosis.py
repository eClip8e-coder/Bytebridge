from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bytebridge.adapters import ByteKVPrefixAdapter, ByteKVPrefixAdapterConfig
from bytebridge.data import encode_bytes_batch, read_jsonl


RUNS = {
    "qwen05_kv_p32_all": {
        "config": "experiments/phase6/kv_prefix_p32_lall/config.yaml",
        "adapter": "experiments/phase6/kv_prefix_p32_lall/adapter.pt",
        "tokenizer_samples": "experiments/phase2/tokenizer_baseline/sample_metrics.jsonl",
        "byte_samples": "experiments/phase6/kv_prefix_p32_lall/eval_test_final/sample_metrics.jsonl",
    },
    "tinyllama_kv_p32_all": {
        "config": "experiments/pricai/tinyllama/kv_prefix_p32_lall/config.yaml",
        "adapter": "experiments/pricai/tinyllama/kv_prefix_p32_lall/adapter.pt",
        "tokenizer_samples": "experiments/pricai/tinyllama/tokenizer_baseline/sample_metrics.jsonl",
        "byte_samples": "experiments/pricai/tinyllama/kv_prefix_p32_lall/eval_test_final/sample_metrics.jsonl",
    },
}


BUCKETS_FOR_GEOMETRY = [
    "clean_english",
    "typo_noise_heavy",
    "unicode_stress",
    "code",
    "tokenizer_stress",
    "multilingual_zh",
    "multilingual_ar",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_lists(cache):
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    keys, values = [], []
    for layer in cache:
        keys.append(layer[0])
        values.append(layer[1])
    return keys, values


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean().sqrt()


def mean_vec(x: torch.Tensor) -> torch.Tensor:
    # [B, H, S, D] -> [D]
    return x.float().mean(dim=(0, 1, 2))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def select_rows(rows: list[dict], max_per_bucket: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["bucket"] in BUCKETS_FOR_GEOMETRY:
            grouped[row["bucket"]].append(row)
    selected = []
    for bucket in BUCKETS_FOR_GEOMETRY:
        selected.extend(grouped[bucket][:max_per_bucket])
    return selected


def load_adapter(cfg: dict, checkpoint_path: Path, device: torch.device) -> ByteKVPrefixAdapter:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    adapter_cfg = ByteKVPrefixAdapterConfig(**ckpt["adapter_config"])
    adapter = ByteKVPrefixAdapter(adapter_cfg).to(device)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    return adapter


def geometry_for_run(repo: Path, run_name: str, run: dict, rows: list[dict], max_per_bucket: int, batch_size: int) -> list[dict]:
    cfg = load_yaml(repo / run["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    adapter = load_adapter(cfg, repo / run["adapter"], device)
    selected = select_rows(rows, max_per_bucket)
    out = []
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            prompts = [sample["input_text"] + str(cfg.get("prompt_suffix", "\n")) for sample in batch]
            by_bucket = [sample["bucket"] for sample in batch]
            byte_ids = encode_bytes_batch(prompts, int(cfg["max_prompt_bytes"])).to(device)
            adapter_cache, _, injected = adapter(byte_ids, llm.config, llm.dtype)
            byte_keys, byte_values = cache_lists(adapter_cache)
            toks = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True, max_length=256, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            native = llm(**toks, use_cache=True)
            native_keys, native_values = cache_lists(native.past_key_values)
            for layer_idx in injected:
                bk = byte_keys[layer_idx]
                bv = byte_values[layer_idx]
                nk = native_keys[layer_idx]
                nv = native_values[layer_idx]
                row_common = {
                    "run": run_name,
                    "layer": layer_idx,
                    "batch_start": start,
                    "buckets": "|".join(sorted(set(by_bucket))),
                    "num_samples": len(batch),
                    "native_seq_len": int(nk.shape[2]),
                    "byte_prefix_len": int(bk.shape[2]),
                }
                out.append(
                    {
                        **row_common,
                        "key_rms_ratio": float(rms(bk) / rms(nk).clamp_min(1e-8)),
                        "value_rms_ratio": float(rms(bv) / rms(nv).clamp_min(1e-8)),
                        "key_mean_cosine": cosine(mean_vec(bk), mean_vec(nk)),
                        "value_mean_cosine": cosine(mean_vec(bv), mean_vec(nv)),
                    }
                )
    del adapter, llm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def summarize_geometry(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run"]].append(row)
    summary = []
    for run, items in grouped.items():
        for key in ["key_rms_ratio", "value_rms_ratio", "key_mean_cosine", "value_mean_cosine"]:
            vals = [x[key] for x in items if not math.isnan(float(x[key]))]
            summary.append(
                {
                    "run": run,
                    "metric": key,
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "n": len(vals),
                }
            )
    return summary


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def case_studies(repo: Path, max_cases: int) -> list[dict]:
    cases = []
    fertility_path = repo / "experiments/pricai/tables/tokenizer_fertility_samples.csv"
    fertility = {}
    if fertility_path.exists():
        with fertility_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fertility[row["id"]] = row
    for run_name, run in RUNS.items():
        tok = {row["id"]: row for row in read_jsonl(repo / run["tokenizer_samples"])}
        byte = {row["id"]: row for row in read_jsonl(repo / run["byte_samples"])}
        by_bucket = defaultdict(list)
        for sample_id, b in byte.items():
            if b["bucket"] not in {"multilingual_zh", "multilingual_ar", "multilingual_ko", "multilingual_ja"}:
                continue
            t = tok.get(sample_id)
            if not t:
                continue
            by_bucket[b["bucket"]].append((float(b["loss"]) - float(t["loss"]), sample_id, t, b))
        for bucket, joined in sorted(by_bucket.items()):
            joined.sort(reverse=True)
            for rank, (_, sample_id, t, b) in enumerate(joined[:max_cases], start=1):
                fert = fertility.get(sample_id, {})
                text = str(b.get("input_text", "")).replace("\n", "\\n")
                target = str(b.get("target_text", "")).replace("\n", "\\n")
                cases.append(
                    {
                        "run": run_name,
                        "rank": rank,
                        "id": sample_id,
                        "bucket": b["bucket"],
                        "tokenizer_loss": float(t["loss"]),
                        "byte_kv_loss": float(b["loss"]),
                        "loss_delta": float(b["loss"]) - float(t["loss"]),
                        "byte_count": int(b.get("byte_count", len(text.encode("utf-8")))),
                        "qwen_input_tokens": fert.get("qwen05_input_tokens", ""),
                        "tinyllama_input_tokens": fert.get("tinyllama_input_tokens", ""),
                        "input_text": text[:220],
                        "target_text": target[:220],
                    }
                )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--max-per-bucket", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=4)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    test_rows = read_jsonl(repo / "data/phase2/splits/test.jsonl")
    geometry = []
    for run_name, run in RUNS.items():
        print(f"Running geometry for {run_name}", flush=True)
        geometry.extend(geometry_for_run(repo, run_name, run, test_rows, args.max_per_bucket, args.batch_size))
    summary = summarize_geometry(geometry)
    cases = case_studies(repo, args.max_cases)
    out_dir = repo / "experiments/pricai/tables"
    write_csv(out_dir / "cross_model_kv_geometry_raw.csv", geometry)
    write_csv(out_dir / "cross_model_kv_geometry_summary.csv", summary)
    write_csv(out_dir / "multilingual_failure_cases.csv", cases)
    with (out_dir / "cross_model_diagnosis_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"geometry_summary": summary, "num_geometry_rows": len(geometry), "num_case_rows": len(cases)}, f, indent=2, ensure_ascii=False)
    print(json.dumps({"geometry_summary": summary, "num_geometry_rows": len(geometry), "num_case_rows": len(cases)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
