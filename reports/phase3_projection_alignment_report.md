# Phase 3 Projection Alignment Report

Date: 2026-05-22

## Goal

Phase 3 tested whether Phase 2 failed because ByteAdapter latents were not
aligned with the frozen Qwen2.5-0.5B input embedding manifold.

The intervention was embedding distillation:

1. Encode clean training text as UTF-8 bytes.
2. Produce ByteAdapter latent embeddings.
3. Tokenize the same text with Qwen tokenizer.
4. Look up frozen Qwen input embeddings.
5. Align byte latents to token-span embeddings plus pooled sequence embeddings.
6. Initialize the normal frozen-LM conditional objective from the distilled
   adapter.

The Phase 2 test manifest and tokenizer baseline were reused unchanged.

## Implemented Artifacts

Scripts:

- `scripts/phase3_train_distill.py`
- `scripts/phase3_eval_compare.py`
- `scripts/phase2_train_bytebridge.py` now supports `init_adapter_checkpoint`

Configs:

- `configs/phase3_distill_l32.yaml`
- `configs/phase3_distill_l64.yaml`
- `configs/phase3_bb_distill_then_lm_l32.yaml`
- `configs/phase3_bb_distill_then_lm_l64.yaml`

Outputs:

- `experiments/phase3/distill_l32_clean/`
- `experiments/phase3/distill_l64_clean/`
- `experiments/phase3/bb_distill_then_lm_l32/`
- `experiments/phase3/bb_distill_then_lm_l64/`
- `experiments/phase3/tables/phase3_comparison.csv`
- `experiments/phase3/tables/phase3_summary.json`

## Distillation Results

| run | latents | initial loss | final loss | loss delta | checksum delta |
|---|---:|---:|---:|---:|---:|
| distill_l32_clean | 32 | 0.9583 | 0.0637 | 0.8946 | 0.0 |
| distill_l64_clean | 64 | 0.9654 | 0.0752 | 0.8902 | 0.0 |

Distillation itself worked: adapter gradients were nonzero, LLM gradients were
zero, and the alignment losses dropped strongly.

## Downstream LM Results

Tokenizer baseline clean loss: 0.3089.

| run | clean loss | clean retention | winning buckets vs tokenizer | checksum delta |
|---|---:|---:|---|---:|
| Phase 2 best: bb_noise_l32_seed1_2k | 0.6770 | 2.19 | typo light/medium/heavy | 0.0 |
| bb_distill_then_lm_l32 | 0.6863 | 2.22 | typo light/medium/heavy | 0.0 |
| bb_distill_then_lm_l64 | 0.7093 | 2.30 | typo light/medium/heavy | 0.0 |

Selected bucket losses:

| bucket | tokenizer | Phase 2 best | distill+LM l32 | distill+LM l64 |
|---|---:|---:|---:|---:|
| clean_english | 0.3089 | 0.6770 | 0.6863 | 0.7093 |
| typo_noise_light | 1.3838 | 0.7454 | 0.7729 | 0.8225 |
| typo_noise_medium | 1.8428 | 0.7786 | 0.8204 | 0.8496 |
| typo_noise_heavy | 2.8353 | 0.8203 | 0.9117 | 0.9905 |
| unicode_stress | 0.5614 | 3.8226 | 4.0054 | 3.9175 |
| code | 0.3876 | 2.4261 | 2.3820 | 2.4878 |
| tokenizer_stress | 0.4806 | 3.1706 | 3.2516 | 3.5280 |
| multilingual_zh | 0.6085 | 4.5419 | 4.4292 | 4.6055 |
| multilingual_ja | 0.5565 | 4.1580 | 3.7644 | 4.9722 |

## Resource Notes

| run | peak allocated MiB | peak reserved MiB | test tokens/sec |
|---|---:|---:|---:|
| bb_distill_then_lm_l32 | 3467.6 | 7868.0 | 24805 |
| bb_distill_then_lm_l64 | 4472.2 | 11428.0 | 19568 |

l64 increased memory and reduced throughput, but did not improve clean
retention.

## Conclusion

Phase 3 is a negative result.

Embedding distillation successfully made the adapter match frozen Qwen input
embeddings under the implemented span/pooled losses, but this did not improve
downstream conditional LM evaluation. Clean retention stayed far above the
hopeful threshold:

- target for hopeful signal: <= 1.5
- target for pass: <= 1.2
- observed best after distillation: 2.22
- Phase 2 best without distillation: 2.19

The typo-noise signal remains, but it did not expand to Unicode, multilingual,
code, or tokenizer-stress buckets.

The current evidence rejects the narrow hypothesis that simple input embedding
manifold alignment is sufficient to fix the Phase 2 failure.

## Interpretation

The failure likely has multiple causes:

1. Matching input embedding geometry is not enough; frozen Qwen may rely on
   token-position and token-boundary structure that fixed byte patches do not
   preserve.
2. Span pooling is a lossy target. It blurs token identity and order inside
   each span.
3. Byte latents act as a prompt prefix in the conditional setup, not as a true
   replacement for the tokenizer path inside Qwen's learned distribution.
4. The adapter is trained only at the input layer; deeper prefix/KV adaptation
   may be required for frozen LMs to use non-token latents reliably.
5. l64 increases context cost but does not solve the semantic alignment problem.

## Recommended Next Routes

Do not scale seeds or ablations on the current fixed-patch method yet.

Better next experiments:

1. Token-level reconstruction pretraining: train byte latents to classify or
   retrieve Qwen token ids before LM training.
2. Boundary-aware dynamic patches: align byte patches to UTF-8/script/token
   boundaries instead of fixed byte windows.
3. Prefix/KV adapter: keep byte latents but inject them into multiple layers,
   closer to prefix tuning.
4. Hybrid bridge: use tokenizer-derived supervision during training, but use
   byte input at inference, with an explicit monotonic alignment module.
5. Decoder-side auxiliary contrastive loss: make each byte latent retrieve its
   corresponding token-span embedding from a batch of negatives.

The negative result should remain part of the project record.
