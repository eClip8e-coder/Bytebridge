# ByteBridge

**ByteBridge** is a diagnostic study of byte-side input adapters for frozen
tokenizer-based language models.

The question is simple:

> Can a frozen language model read raw UTF-8 byte input if we train only a small
> external adapter?

The answer from this project is mixed. Byte adapters can help on some
tokenizer-unfriendly inputs, especially typo noise, but they do not reliably
replace native tokenization. The strongest result is model-dependent: Qwen2.5
remains below its tokenizer baseline outside typo-noise buckets, while
TinyLlama all-layer KV-prefix injection is stronger on clean, typo, and Unicode
inputs but still fails on multilingual and tokenizer-stress buckets.

This repository contains the code, fixed benchmark splits, compact metrics, and
paper artifacts behind that study. Large adapter checkpoints are intentionally
excluded.

## Main Findings

- **Input-layer byte replacement is not enough.** Fixed byte patches through
  `inputs_embeds` train successfully, but clean-text loss remains far worse than
  native tokenized input.
- **Embedding alignment is not enough.** Distilling byte latents toward token
  embeddings reduces the auxiliary loss but does not improve downstream LM
  evaluation.
- **Token reconstruction helps, but does not solve the problem.** Byte spans can
  recover tokenizer token identity, especially with oracle token boundaries, but
  Qwen2.5 still does not use the resulting input-layer latents like native
  token embeddings.
- **Deeper injection helps.** Soft prefixes and KV-prefixes improve several
  stress buckets, but Qwen2.5 still misses the clean-retention threshold.
- **The result is model- and bucket-dependent.** TinyLlama benefits much more
  from all-layer KV-prefix injection than Qwen2.5, while still failing badly on
  multilingual and tokenizer-stress inputs.

## Repository Layout

```text
bytebridge/              Core adapter, data, mapping, and metric code
scripts/                 Training, evaluation, comparison, and plotting scripts
configs/                 YAML configs for all reported runs
data/phase2/splits/      Fixed train/validation/test manifests
experiments/             Compact metrics, summaries, and result tables
reports/                 Phase-by-phase technical reports
paper/pricai2026/        PRICAI paper source and figures
paper/tokshop2026/       Earlier short workshop version
```

The `experiments/` directory in this repository contains small JSON/CSV/JSONL
artifacts needed for auditing results. It does **not** include trained adapter
checkpoints such as `adapter.pt`, `best_adapter.pt`, or `final_adapter.pt`.

## Benchmark

The fixed benchmark uses paired conditional evaluation:

- `input_text` is the prompt.
- `target_text` is the supervised suffix.
- Noisy examples use noisy input and clean target text.
- Tokenizer baselines and ByteBridge adapters are scored on the same target
  tokens.

Buckets include:

- clean English
- light/medium/heavy typo noise
- Unicode stress
- multilingual short text
- code snippets
- tokenizer-stress strings such as URLs, paths, IDs, hex/base64-like strings,
  and logs

The fixed manifests live in `data/phase2/splits/`.

## Key Results

Representative Qwen2.5-0.5B results:

| Run | Clean retention | Wins vs tokenizer |
|---|---:|---|
| Fixed input byte adapter | 2.19x | typo light / medium / heavy |
| Distill then LM | 2.22x | typo light / medium / heavy |
| Token reconstruction then LM | 1.84x | typo light / medium / heavy |
| Soft prefix p64 | 1.76x | typo light / medium / heavy |
| KV prefix p32 all layers | 1.61x | typo light / medium / heavy |

Selected cross-model result:

| Model | Best selected interface | Clean retention | Main outcome |
|---|---|---:|---|
| Qwen2.5-1.5B | KV prefix p32 all layers | 2.52x | still negative |
| TinyLlama-1.1B-Chat | KV prefix p32 all layers | 0.63x | wins clean, typo, Unicode; loses multilingual and tokenizer-stress |

For full per-bucket numbers, see:

- `reports/phase2_research_report.md`
- `reports/phase3_projection_alignment_report.md`
- `reports/phase4_boundary_reconstruction_report.md`
- `reports/phase5_prefix_kv_injection_report.md`
- `reports/phase6_kv_prefix_report.md`
- `reports/pricai_enhancement_notes.md`
- `experiments/pricai/tables/`

## Reproducing the Pipeline

Install the usual PyTorch/Hugging Face stack for your GPU environment, then run
commands from the repository root.

Build the fixed data splits:

```bash
python scripts/phase2_build_data.py --config configs/phase2_data.yaml
```

Evaluate the native tokenizer baseline:

```bash
python scripts/phase2_eval_tokenizer_baseline.py \
  --config configs/phase2_tokenizer_baseline.yaml
```

Run a representative byte adapter experiment:

```bash
python scripts/phase2_train_bytebridge.py \
  --config configs/phase2_bb_noise_l32_seed1_2k.yaml
```

Run the later KV-prefix diagnostic:

```bash
python scripts/phase6_probe_qwen_kv_cache.py
python scripts/phase6_train_kv_prefix.py \
  --config configs/phase6_kv_prefix_p32_lall.yaml
```

Run the PRICAI cross-model comparison scripts:

```bash
python scripts/pricai_compare_tokenizer_fertility.py
python scripts/pricai_compare_training_curves.py
python scripts/pricai_cross_model_diagnosis.py
```

Many scripts expect Hugging Face model access for:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-1.5B`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Paper

The current PRICAI paper source is in `paper/pricai2026/`.

Important note: the paper in this private repository is an anonymous submission
artifact. Do not make the repository public during double-anonymous review.

## What Is Not Included

To keep the repository lightweight and review-safe, the following are excluded:

- virtual environments
- downloaded model weights
- adapter checkpoints
- large binary archives
- LaTeX build intermediates

The compact metrics in `experiments/` are enough to audit the reported tables and
figures. Checkpoints can be regenerated from the configs and scripts.

## Citation

No public citation is provided while the paper is under anonymous review.

