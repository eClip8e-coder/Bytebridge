from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def bucket_counts(rows: Iterable[dict]) -> dict[str, int]:
    return dict(Counter(row["bucket"] for row in rows))


def check_clean_id_leakage(train_rows: list[dict], test_rows: list[dict]) -> dict:
    train_ids = {r["clean_id"] for r in train_rows if r.get("clean_id")}
    test_ids = {r["clean_id"] for r in test_rows if r.get("clean_id")}
    overlap = sorted(train_ids & test_ids)
    return {"overlap_count": len(overlap), "overlap_examples": overlap[:20]}
