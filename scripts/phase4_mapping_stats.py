from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from transformers import AutoTokenizer

from bytebridge.data import read_jsonl, write_jsonl
from bytebridge.data.token_byte_mapping import heuristic_byte_spans, token_byte_spans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_mapping_stats.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows = read_jsonl(cfg["manifest_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    per_bucket = defaultdict(lambda: {"samples": 0, "tokens": 0, "unmapped": 0, "token_bytes": 0, "heuristic_spans": 0})
    examples = []
    for sample in rows:
        text = sample["input_text"] + str(cfg.get("prompt_suffix", "\n"))
        spans, meta = token_byte_spans(text, tokenizer)
        hspans = heuristic_byte_spans(
            text,
            max_spans=int(cfg["max_spans"]),
            max_bytes=int(cfg["max_prompt_bytes"]),
            max_span_bytes=int(cfg["heuristic_max_span_bytes"]),
        )
        bucket = sample["bucket"]
        mapped = [s for s in spans if s.mapped and s.byte_end > s.byte_start]
        per_bucket[bucket]["samples"] += 1
        per_bucket[bucket]["tokens"] += meta["total"]
        per_bucket[bucket]["unmapped"] += meta["unmapped"]
        per_bucket[bucket]["token_bytes"] += sum(s.byte_end - s.byte_start for s in mapped)
        per_bucket[bucket]["heuristic_spans"] += len(hspans)
        if len(examples) < int(cfg["num_examples"]):
            examples.append(
                {
                    "id": sample["id"],
                    "bucket": bucket,
                    "text": sample["input_text"],
                    "mapping_method": meta["method"],
                    "unmapped": meta["unmapped"],
                    "tokens": [
                        {
                            "token_id": s.token_id,
                            "token_text": s.token_text,
                            "byte_start": s.byte_start,
                            "byte_end": s.byte_end,
                            "mapped": s.mapped,
                        }
                        for s in spans[: int(cfg["max_example_tokens"])]
                    ],
                    "heuristic_spans": hspans[: int(cfg["max_example_tokens"])],
                }
            )

    summary = {}
    for bucket, stats in sorted(per_bucket.items()):
        tokens = max(1, stats["tokens"])
        samples = max(1, stats["samples"])
        mapped_tokens = max(1, stats["tokens"] - stats["unmapped"])
        summary[bucket] = {
            **stats,
            "unmapped_rate": stats["unmapped"] / tokens,
            "avg_token_bytes": stats["token_bytes"] / mapped_tokens,
            "avg_heuristic_spans": stats["heuristic_spans"] / samples,
        }
    with open(out_dir / "mapping_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_jsonl(out_dir / "mapping_examples.jsonl", examples)
    print(json.dumps({"output_dir": str(out_dir), "buckets": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
