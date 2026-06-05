from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def cache_summary(cache) -> dict:
    out = {"type": type(cache).__name__}
    layers = []
    if hasattr(cache, "layers"):
        out["num_layers"] = len(cache.layers)
        for i, layer in enumerate(cache.layers):
            entry = {"layer": i, "type": type(layer).__name__}
            for attr in ["keys", "values", "key_cache", "value_cache"]:
                if hasattr(layer, attr):
                    val = getattr(layer, attr)
                    if torch.is_tensor(val):
                        entry[attr] = list(val.shape)
                    elif isinstance(val, list):
                        entry[attr] = [list(x.shape) if torch.is_tensor(x) else str(type(x)) for x in val]
            layers.append(entry)
    elif isinstance(cache, (tuple, list)):
        out["num_layers"] = len(cache)
        for i, item in enumerate(cache):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                layers.append({"layer": i, "key_shape": list(item[0].shape), "value_shape": list(item[1].shape)})
    out["layers"] = layers[:4]
    return out


def make_dynamic_prefix(config, batch_size: int, prefix_length: int, dtype, device) -> DynamicCache:
    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    cache = DynamicCache(config=config)
    for layer_idx in range(num_layers):
        key = torch.zeros(batch_size, num_kv_heads, prefix_length, head_dim, dtype=dtype, device=device)
        value = torch.zeros(batch_size, num_kv_heads, prefix_length, head_dim, dtype=dtype, device=device)
        cache.update(key, value, layer_idx)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", default="experiments/phase6/kv_probe")
    parser.add_argument("--text", default="ByteBridge KV cache probe.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    encoded = tokenizer(args.text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded, use_cache=True)
    past = outputs.past_key_values
    probe = {
        "model_name": args.model_name,
        "config": {
            "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
            "head_dim": getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads),
        },
        "normal_cache": cache_summary(past),
        "manual_dynamic_cache_forward": {"ok": False},
        "manual_legacy_tuple_forward": {"ok": False},
    }

    input_ids = encoded["input_ids"]
    target = input_ids[:, :2]
    attention_mask = torch.ones((target.shape[0], 2 + target.shape[1]), dtype=torch.long, device=device)
    try:
        cache = make_dynamic_prefix(model.config, target.shape[0], 2, model.dtype, device)
        with torch.no_grad():
            out = model(input_ids=target, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
        probe["manual_dynamic_cache_forward"] = {
            "ok": True,
            "logits_shape": list(out.logits.shape),
            "output_cache": cache_summary(out.past_key_values),
        }
    except Exception as exc:  # noqa: BLE001
        probe["manual_dynamic_cache_forward"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        num_layers = int(model.config.num_hidden_layers)
        num_kv_heads = int(getattr(model.config, "num_key_value_heads", model.config.num_attention_heads))
        head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
        legacy = tuple(
            (
                torch.zeros(target.shape[0], num_kv_heads, 2, head_dim, dtype=model.dtype, device=device),
                torch.zeros(target.shape[0], num_kv_heads, 2, head_dim, dtype=model.dtype, device=device),
            )
            for _ in range(num_layers)
        )
        with torch.no_grad():
            out = model(input_ids=target, attention_mask=attention_mask, past_key_values=legacy, use_cache=True)
        probe["manual_legacy_tuple_forward"] = {
            "ok": True,
            "logits_shape": list(out.logits.shape),
            "output_cache": cache_summary(out.past_key_values),
        }
    except Exception as exc:  # noqa: BLE001
        probe["manual_legacy_tuple_forward"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_cuda_version": torch.version.cuda,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(probe, f, indent=2, ensure_ascii=False)
    with open(out_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
    print(json.dumps(probe, ensure_ascii=False))


if __name__ == "__main__":
    main()
