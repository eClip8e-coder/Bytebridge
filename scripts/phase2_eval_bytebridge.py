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
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from bytebridge.adapters import ByteAdapter, ByteAdapterConfig
from bytebridge.data import read_jsonl
from scripts.phase2_train_bytebridge import checksum_first_tensors, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split-name", default="test")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.manifest:
        cfg["test_manifest_path"] = args.manifest
    out_dir = Path(args.output_dir or Path(args.checkpoint).resolve().parent / "standalone_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

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
    initial_checksum = checksum_first_tensors(llm)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    adapter_cfg = ByteAdapterConfig(**ckpt["adapter_config"])
    adapter = ByteAdapter(adapter_cfg).to(device)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    rows = read_jsonl(cfg["test_manifest_path"])
    result = evaluate(adapter, llm, tokenizer, rows, cfg, device, out_dir, args.split_name)
    summary = {
        "checkpoint": args.checkpoint,
        "manifest": cfg["test_manifest_path"],
        "result": result,
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "llm_trainable_params": sum(p.numel() for p in llm.parameters() if p.requires_grad),
        "llm_checksum_delta_first8_tensors": checksum_first_tensors(llm) - initial_checksum,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
