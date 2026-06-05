# ByteBridge

Tokenizer-free byte input adapters for frozen causal language models.

This repository is a reproducible research prototype for testing whether small
byte-side adapters can provide input context to frozen tokenizer-based causal
language models. The project starts with an `inputs_embeds` feasibility gate and
then builds a diagnostic ladder: tokenizer baselines, noisy and tokenizer-stress
benchmarks, embedding distillation, token reconstruction, soft-prefix injection,
KV-prefix injection, and selected cross-model replications.

Large adapter checkpoints are intentionally not stored in this repository.
Configuration files, scripts, fixed data splits, reports, and compact metrics are
included so that experiments can be reproduced or audited.

## Feasibility Gate

```bash
cd bytebridge
python3 scripts/feasibility_gate.py --config configs/feasibility_qwen05.yaml
```

Outputs are written to `experiments/feasibility/qwen05/`:

- `metrics.csv`: per-step loss, gradient checks, memory, elapsed time
- `env.json`: Python/PyTorch/CUDA/GPU/model/config metadata
- `summary.json`: pass/fail-relevant summary values
- `adapter.pt`: adapter checkpoint
- `loss_curve.png`: training loss plot

Current result with `Qwen/Qwen2.5-0.5B`: loss decreased from 12.6508 to 4.3031
over 120 steps, adapter gradients were nonzero, LLM gradients remained zero, and
the first-four-tensor LLM checksum delta was 0.0.

The environment package list used for the first run is saved at
`experiments/feasibility/requirements_user.txt`. This machine did not have
`python3-venv`, `pip`, `conda`, or sudo access initially, so user-level `pip` was
bootstrapped with `get-pip.py --user --break-system-packages`.

## Scope

The repository validates:

- `inputs_embeds` forward/backward on a frozen causal LM
- no trainable LLM parameters
- nonzero adapter gradients
- decreasing adapter training loss
- recorded GPU memory, batch size, latent length, and runtime
- fixed train/validation/test splits for clean, noisy, Unicode, multilingual,
  code, and tokenizer-stress buckets
- per-bucket comparison against native tokenizer baselines

## Phase 2 Pilot

Phase 2 uses paired conditional evaluation: `input_text` is the prompt and
`target_text` is the supervised suffix. Noisy examples use noisy input and clean
target text. Tokenizer baseline and ByteBridge are scored on the same target
tokens.

```bash
python3 scripts/phase2_build_data.py --config configs/phase2_data.yaml
python3 scripts/phase2_eval_tokenizer_baseline.py --config configs/phase2_tokenizer_baseline.yaml
python3 scripts/phase2_train_bytebridge.py --config configs/phase2_bb_noise_l32_seed1_2k.yaml
```

Current Phase 2 pilot result: partial/negative. ByteBridge wins on typo-noise
buckets but fails clean retention and remains worse on Unicode, multilingual,
code, and tokenizer-stress buckets. See
`reports/phase2_research_report.md`.

## Phase 3 Projection Alignment

Phase 3 tested embedding distillation before LM training:

```bash
python3 scripts/phase3_train_distill.py --config configs/phase3_distill_l32.yaml
python3 scripts/phase2_train_bytebridge.py --config configs/phase3_bb_distill_then_lm_l32.yaml
python3 scripts/phase3_eval_compare.py --phase3-runs experiments/phase3/bb_distill_then_lm_l32/summary.json experiments/phase3/bb_distill_then_lm_l64/summary.json
```

Current Phase 3 result: negative. Distillation loss dropped strongly, but clean
retention did not improve versus the Phase 2 best run. See
`reports/phase3_projection_alignment_report.md`.

## Phase 4 Boundary Reconstruction

Phase 4 tests token-byte mapping, boundary-aware spans, and token reconstruction
pretraining:

```bash
python3 scripts/phase4_mapping_stats.py --config configs/phase4_mapping_stats.yaml
python3 scripts/phase4_train_token_reconstruction.py --config configs/phase4_recon_oracle_boundary_l32.yaml
python3 scripts/phase4_train_after_reconstruction.py --config configs/phase4_bb_recon_oracle_boundary_l32_then_lm.yaml
python3 scripts/phase4_compare.py --runs experiments/phase2/bb_noise_l32_seed1_2k/summary.json experiments/phase3/bb_distill_then_lm_l32/summary.json experiments/phase4/bb_recon_fixed_l32_then_lm/summary.json experiments/phase4/bb_recon_oracle_boundary_l32_then_lm/summary.json experiments/phase4/bb_recon_heuristic_boundary_l32_then_lm/summary.json
```

Current Phase 4 result: diagnostic partial. Token reconstruction works,
especially with oracle token boundaries, and fixed reconstruction improves clean
retention from `2.19x` to `1.84x`. It still does not reach the `<=1.5x` hopeful
threshold or beat tokenizer baseline outside typo buckets. See
`reports/phase4_boundary_reconstruction_report.md`.

## Phase 5 Soft Prefix Injection

Phase 5A tests byte-derived soft prefix embeddings:

```bash
python3 scripts/phase5_train_prefix.py --config configs/phase5_prefix_p16.yaml
python3 scripts/phase5_train_prefix.py --config configs/phase5_prefix_p32.yaml
python3 scripts/phase5_train_prefix.py --config configs/phase5_prefix_p64.yaml
python3 scripts/phase5_compare.py --runs experiments/phase2/bb_noise_l32_seed1_2k/summary.json experiments/phase3/bb_distill_then_lm_l32/summary.json experiments/phase4/bb_recon_fixed_l32_then_lm/summary.json experiments/phase5/prefix_fixed_p16/summary.json experiments/phase5/prefix_fixed_p32/summary.json experiments/phase5/prefix_fixed_p64/summary.json
```

Current Phase 5A result: diagnostic partial. p64 improves code, Unicode, and
tokenizer-stress substantially versus prior ByteBridge variants, but best clean
retention is still `1.76x`, above the `<=1.5x` hopeful threshold. See
`reports/phase5_prefix_kv_injection_report.md`.

## Phase 6 KV-Prefix Injection

Phase 6 tests byte-derived `DynamicCache` KV-prefix injection for frozen Qwen:

```bash
python3 scripts/phase6_probe_qwen_kv_cache.py
python3 scripts/phase6_train_kv_prefix.py --config configs/phase6_kv_prefix_p16_l4.yaml
python3 scripts/phase6_train_kv_prefix.py --config configs/phase6_kv_prefix_p32_l4.yaml
python3 scripts/phase6_train_kv_prefix.py --config configs/phase6_kv_prefix_p32_lall.yaml
python3 scripts/phase6_compare.py --runs experiments/phase2/bb_noise_l32_seed1_2k/summary.json experiments/phase4/bb_recon_fixed_l32_then_lm/summary.json experiments/phase5/prefix_fixed_p64/summary.json experiments/phase6/kv_prefix_p16_l4/summary.json experiments/phase6/kv_prefix_p32_l4/summary.json experiments/phase6/kv_prefix_p32_lall/summary.json
```

Current Phase 6 result: diagnostic partial / negative. Qwen `DynamicCache`
manual KV-prefix injection works, and all-layer KV prefix improves clean
retention from Phase 5 p64 `1.76x` to `1.61x`. It still misses the `<=1.5x`
hopeful threshold and does not beat the tokenizer baseline outside typo-noise
buckets. See `reports/phase6_kv_prefix_report.md`.

## PRICAI Extension

The PRICAI extension adds selected replications on `Qwen/Qwen2.5-1.5B` and
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, tokenizer fragmentation analysis,
training-curve comparisons, and multilingual failure cases. The main finding is
mixed: Qwen2.5 variants remain below the native tokenizer baseline outside
typo-noise buckets, while TinyLlama all-layer KV-prefix injection is stronger on
clean, typo, and Unicode buckets but still fails on multilingual and
tokenizer-stress inputs.

The paper source is in `paper/pricai2026/`; compact PRICAI result tables are in
`experiments/pricai/tables/`.
