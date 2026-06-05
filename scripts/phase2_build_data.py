from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from transformers import AutoTokenizer

from bytebridge.data import bucket_counts, check_clean_id_leakage, write_jsonl
from bytebridge.data.perturbations import normalize_variant, perturb_text


EN_SUBJECTS = [
    "research prototypes",
    "language models",
    "tokenizers",
    "small datasets",
    "unicode strings",
    "debug logs",
    "training scripts",
    "evaluation metrics",
    "software tests",
    "frozen models",
]
EN_VERBS = [
    "need careful baselines",
    "can fail under noise",
    "should record every seed",
    "often hide edge cases",
    "benefit from simple checks",
    "must avoid data leakage",
    "produce useful negative results",
    "require reproducible commands",
]
EN_CONTEXTS = [
    "The experiment uses a fixed train and test split.",
    "A clear metric is better than a vague claim.",
    "The adapter receives bytes and emits latent vectors.",
    "The baseline keeps the original tokenizer path.",
    "Every result should be grouped by input bucket.",
    "A held out set makes failure visible.",
]


UNICODE_TEXTS = [
    "cafe\u0301 and café should both be measured.",
    "hello\u200dworld contains a zero width joiner.",
    "ＡＢＣ１２３ uses full-width characters.",
    "price is €１２.５０ ✅ and tax is ¥300.",
    "math symbols: ∑ᵢ xᵢ → ∞, αβγ, √2 ≈ 1.414.",
    "family emoji: 👨‍👩‍👧‍👦, flags: 🇺🇳 🇯🇵, skin tones: 👍🏽.",
    "quotes vary: “smart”, 'plain', «guillemets».",
    "combining marks: a̐éö̲ and normalized forms differ.",
]

MULTILINGUAL_TEXTS = {
    "zh": [
        "这个实验需要清晰的基线和可复现的结果。",
        "字节级输入可以绕过固定分词器。",
        "模型主体被冻结，只训练输入适配器。",
    ],
    "ja": [
        "この評価ではノイズに対する頑健性を測定します。",
        "トークナイザーを使わずにバイト列を入力します。",
        "小さな実験でも失敗条件を明確にします。",
    ],
    "ko": [
        "이 실험은 작은 모델로 재현 가능해야 합니다.",
        "바이트 입력 어댑터만 학습합니다.",
        "평가는 버킷별로 나누어 기록합니다.",
    ],
    "ar": [
        "يجب أن تكون النتائج قابلة للتكرار وواضحة.",
        "يتم تجميد نموذج اللغة وتدريب المحول فقط.",
        "النصوص متعددة اللغات تكشف تحيزات التقسيم.",
    ],
    "hi": [
        "यह प्रयोग छोटे मॉडल पर दोहराया जा सकता है।",
        "बाइट इनपुट से टोकनाइज़र की निर्भरता घटती है।",
        "हर बकेट के लिए अलग मीट्रिक चाहिए।",
    ],
    "ru": [
        "Эксперимент должен иметь честные базовые линии.",
        "Адаптер байтов обучается при замороженной модели.",
        "Юникод и шум проверяют устойчивость токенизации.",
    ],
    "latin": [
        "El modelo debe manejar acentos y señales mixtas.",
        "Le résumé contient des caractères français.",
        "Die Prüfung nutzt Umlaute wie ä, ö und ü.",
    ],
}

CODE_TEXTS = [
    "def add(x, y):\n    return x + y",
    "const value = items?.[0] ?? 42;",
    "{\"status\":\"ok\",\"items\":[1,2,3],\"path\":\"/tmp/a b\"}",
    "grep -R \"ByteBridge\" ./reports | head -n 5",
    "ValueError: expected shape (4, 32, 896), got (4, 31, 896)",
    r"regex = r'^[A-Za-z0-9_\-]+@[a-z]+\.[a-z]{2,}$'",
    "SELECT id, name FROM users WHERE active = TRUE ORDER BY id DESC;",
    "CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --steps 2000",
]

STRESS_TEXTS = [
    "/usr/local/cuda-12.8/bin/nvcc",
    "https://example.com/a/b?q=hello%20world&id=9f2a",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
    "0x9f2a7cbb01eeff",
    "550e8400-e29b-41d4-a716-446655440000",
    "byte_bridge_adapter_v2/checkpoint-000250.safetensors",
    "camelCaseIdentifierWithHTTPAndJSONMix",
    "kebab-case-package-name==1.2.3",
    "sha256:9f86d081884c7d659a2feaa0c55ad015",
    "user.name+tag@example-domain.co.uk",
]


def make_clean_text(i: int, rng: random.Random) -> str:
    subject = rng.choice(EN_SUBJECTS)
    verb = rng.choice(EN_VERBS)
    context = rng.choice(EN_CONTEXTS)
    suffix = rng.choice(
        [
            f"Sample {i} keeps the wording compact.",
            f"The run id is bb-{i:05d}.",
            f"It includes number {rng.randrange(10, 9999)} for variety.",
            "Short contexts make single GPU evaluation cheap.",
        ]
    )
    return f"{subject.capitalize()} {verb}. {context} {suffix}"


def row(
    split: str,
    bucket: str,
    idx: int,
    input_text: str,
    source: str,
    seed: int,
    perturbation_type: str = "none",
    noise_level: str = "none",
    clean_id: str | None = None,
    target_text: str | None = None,
) -> dict:
    sample_id = f"{split}_{bucket}_{idx:06d}"
    target = target_text if target_text is not None else input_text
    clean_hash = hashlib.sha256(target.encode("utf-8")).hexdigest()
    return {
        "id": sample_id,
        "bucket": bucket,
        "text": input_text,
        "input_text": input_text,
        "target_text": target,
        "source": source,
        "perturbation_type": perturbation_type,
        "noise_level": noise_level,
        "seed": seed,
        "clean_id": clean_id or sample_id,
        "clean_sha256": clean_hash,
        "source_id": clean_id or sample_id,
        "noise_recipe_id": f"{perturbation_type}:{noise_level}:{seed}",
    }


def build_clean(split: str, count: int, seed: int, offset: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        rows.append(row(split, "clean_english", offset + i, make_clean_text(offset + i, rng), "synthetic_templates", seed))
    return rows


def build_noise(split: str, clean_rows: list[dict], seed: int) -> list[dict]:
    rows = []
    perturbations = ["delete", "insert", "swap", "substitute", "repeat", "whitespace", "case"]
    levels = [("light", 0.05), ("medium", 0.10), ("heavy", 0.20)]
    idx = 0
    for clean in clean_rows:
        for level_name, level in levels:
            ptype = perturbations[(idx + seed) % len(perturbations)]
            pseed = seed + idx * 17
            text = perturb_text(clean["text"], ptype, level, pseed)
            rows.append(
                row(
                    split,
                    f"typo_noise_{level_name}",
                    idx,
                    text,
                    "synthetic_perturbation",
                    pseed,
                    ptype,
                    level_name,
                    clean["clean_id"],
                    target_text=clean["target_text"],
                )
            )
            idx += 1
    return rows


def cycle_rows(split: str, bucket: str, texts: list[str], count: int, seed: int, source: str) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        base = texts[i % len(texts)]
        if count > len(texts):
            base = f"{base} [{rng.randrange(1000, 9999)}]"
        rows.append(row(split, bucket, i, base, source, seed + i))
    return rows


def multilingual_rows(split: str, count_per_group: int, seed: int) -> list[dict]:
    rows = []
    for lang, texts in MULTILINGUAL_TEXTS.items():
        bucket = f"multilingual_{lang}"
        rows.extend(cycle_rows(split, bucket, texts, count_per_group, seed + len(rows), f"curated_{lang}"))
    return rows


def unicode_rows(split: str, count: int, seed: int) -> list[dict]:
    rows = cycle_rows(split, "unicode_stress", UNICODE_TEXTS, count, seed, "curated_unicode")
    for i, base in enumerate(UNICODE_TEXTS[: min(len(UNICODE_TEXTS), count // 4)]):
        text = normalize_variant(base, "NFD" if i % 2 == 0 else "NFC")
        rows.append(row(split, "unicode_stress", count + i, text, "unicode_normalization", seed + i, "normalization", "mixed"))
    return rows


def add_token_stats(rows: list[dict], tokenizer, max_length: int) -> None:
    for sample in rows:
        input_ids = tokenizer(sample["input_text"], truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(sample["target_text"], truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
        sample["input_byte_len"] = len(sample["input_text"].encode("utf-8"))
        sample["target_byte_len"] = len(sample["target_text"].encode("utf-8"))
        sample["input_token_len"] = len(input_ids)
        sample["target_token_len"] = len(target_ids)
        sample["byte_len"] = sample["input_byte_len"]
        sample["token_len"] = sample["input_token_len"]


def summarize(rows_by_split: dict[str, list[dict]], leakage: dict, out_path: Path) -> None:
    summary = {"splits": {}, "leakage_train_test": leakage}
    for split, rows in rows_by_split.items():
        summary["splits"][split] = {
            "count": len(rows),
            "bucket_counts": bucket_counts(rows),
            "mean_byte_len": sum(r["byte_len"] for r in rows) / len(rows),
            "mean_token_len": sum(r["token_len"] for r in rows) / len(rows),
            "mean_target_token_len": sum(r["target_token_len"] for r in rows) / len(rows),
            "max_byte_len": max(r["byte_len"] for r in rows),
            "max_token_len": max(r["token_len"] for r in rows),
            "max_target_token_len": max(r["target_token_len"] for r in rows),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_data.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    max_length = int(cfg["max_token_length"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_clean = build_clean("train", int(cfg["train_clean_count"]), seed + 1, 0)
    val_clean = build_clean("validation", int(cfg["val_clean_count"]), seed + 2, 100_000)
    test_clean = build_clean("test", int(cfg["test_clean_count"]), seed + 3, 200_000)

    train = list(train_clean)
    if cfg.get("include_train_noise", True):
        train.extend(build_noise("train", train_clean[: int(cfg["train_noise_base_count"])], seed + 10))
    train.extend(unicode_rows("train", int(cfg["train_unicode_count"]), seed + 20))
    train.extend(cycle_rows("train", "code", CODE_TEXTS, int(cfg["train_code_count"]), seed + 30, "curated_code"))
    train.extend(cycle_rows("train", "tokenizer_stress", STRESS_TEXTS, int(cfg["train_stress_count"]), seed + 40, "curated_stress"))

    validation = list(val_clean)
    validation.extend(build_noise("validation", val_clean[: int(cfg["val_noise_base_count"])], seed + 50))
    validation.extend(unicode_rows("validation", int(cfg["val_unicode_count"]), seed + 60))
    validation.extend(multilingual_rows("validation", int(cfg["val_multilingual_per_group"]), seed + 70))
    validation.extend(cycle_rows("validation", "code", CODE_TEXTS, int(cfg["val_code_count"]), seed + 80, "curated_code"))
    validation.extend(cycle_rows("validation", "tokenizer_stress", STRESS_TEXTS, int(cfg["val_stress_count"]), seed + 90, "curated_stress"))

    test = list(test_clean)
    test.extend(build_noise("test", test_clean[: int(cfg["test_noise_base_count"])], seed + 100))
    test.extend(unicode_rows("test", int(cfg["test_unicode_count"]), seed + 110))
    test.extend(multilingual_rows("test", int(cfg["test_multilingual_per_group"]), seed + 120))
    test.extend(cycle_rows("test", "code", CODE_TEXTS, int(cfg["test_code_count"]), seed + 130, "curated_code"))
    test.extend(cycle_rows("test", "tokenizer_stress", STRESS_TEXTS, int(cfg["test_stress_count"]), seed + 140, "curated_stress"))

    rows_by_split = {"train": train, "validation": validation, "test": test}
    for rows in rows_by_split.values():
        add_token_stats(rows, tokenizer, max_length)

    out_dir = Path(cfg["output_dir"])
    for split, rows in rows_by_split.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)

    leakage = check_clean_id_leakage(train, test)
    summarize(rows_by_split, leakage, Path(cfg["summary_path"]))
    print(json.dumps({"output_dir": str(out_dir), "leakage": leakage, "test_buckets": bucket_counts(test)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
