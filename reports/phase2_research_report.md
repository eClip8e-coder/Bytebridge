# Phase 2 Preliminary Research Report

Date: 2026-05-22

## Summary

Phase 2 is currently a negative/partial result.

The evaluation harness, fixed manifests, tokenizer baseline, ByteBridge
clean-only run, and ByteBridge noise-augmented runs are implemented. The frozen
LLM checks pass: Qwen2.5-0.5B remains frozen, LLM trainable parameters are 0,
and checksum delta is 0.0 in ByteBridge runs.

However, the current ByteBridge l32 adapter does not meet the Phase 2 success
criteria. It improves loss on typo-noise buckets, but clean retention is far
worse than the allowed threshold and Unicode/multilingual/code/tokenizer-stress
buckets remain much worse than the tokenizer baseline.

## Evaluation Setup

Model: `Qwen/Qwen2.5-0.5B`

Task framing:

- Tokenizer baseline: tokenize `input_text` as prompt, append `target_text`, and
  compute loss only on target tokens.
- ByteBridge: encode `input_text` as UTF-8 bytes, map to 32 prompt latents,
  append frozen Qwen embeddings for `target_text`, and compute loss only on the
  same target tokens.
- Noisy examples use noisy `input_text` and clean `target_text`.
- Clean and stress examples use identical input and target text.

This avoids the Phase 1 limitation where labels were truncated to the number of
byte patches.

## Implemented Commands

```bash
python3 scripts/phase2_build_data.py --config configs/phase2_data.yaml
python3 scripts/phase2_eval_tokenizer_baseline.py --config configs/phase2_tokenizer_baseline.yaml
python3 scripts/phase2_train_bytebridge.py --config configs/phase2_bb_clean_l32_seed1.yaml
python3 scripts/phase2_train_bytebridge.py --config configs/phase2_bb_noise_l32_seed1.yaml
python3 scripts/phase2_train_bytebridge.py --config configs/phase2_bb_noise_l32_seed1_2k.yaml
python3 scripts/phase2_compare_runs.py --runs experiments/phase2/bb_clean_l32_seed1/summary.json experiments/phase2/bb_noise_l32_seed1/summary.json experiments/phase2/bb_noise_l32_seed1_2k/summary.json
```

## Key Results

Tokenizer baseline clean loss: 0.3089.

| run | steps | clean loss | clean retention | winning buckets vs tokenizer |
|---|---:|---:|---:|---|
| bb_clean_l32_seed1 | 800 | 0.8023 | 2.60 | typo light, typo medium, typo heavy |
| bb_noise_l32_seed1 | 800 | 0.8803 | 2.85 | typo light, typo medium, typo heavy |
| bb_noise_l32_seed1_2k | 2000 | 0.6770 | 2.19 | typo light, typo medium, typo heavy |

Selected bucket losses:

| bucket | tokenizer | bb_noise_l32_2k |
|---|---:|---:|
| clean_english | 0.3089 | 0.6770 |
| typo_noise_light | 1.3838 | 0.7454 |
| typo_noise_medium | 1.8428 | 0.7786 |
| typo_noise_heavy | 2.8353 | 0.8203 |
| unicode_stress | 0.5614 | 3.8226 |
| code | 0.3876 | 2.4261 |
| tokenizer_stress | 0.4806 | 3.1706 |
| multilingual_zh | 0.6085 | 4.5419 |
| multilingual_ja | 0.5565 | 4.1580 |

## Pass/Fail Against Phase 2 Criteria

Passed:

- Tokenizer baseline implemented and evaluated on all buckets.
- Fixed train/validation/test manifests are written to disk.
- Train/test `clean_id` leakage check reports 0 overlap.
- ByteBridge trains with nonzero adapter gradients.
- LLM and LM head remain frozen; checksum delta is 0.0.
- Metrics are reported per bucket, not only as a global average.

Failed:

- Clean retention threshold fails badly. Best current run is 2.19, while the
  target is <= 1.2.
- ByteBridge does not beat tokenizer baseline on Unicode, multilingual, code, or
  tokenizer-stress buckets.
- Noise augmentation helps compared with clean-only on code/stress buckets, but
  not enough to be competitive with the tokenizer baseline.

Partial signal:

- ByteBridge l32 consistently beats tokenizer baseline on typo-noise buckets in
  this conditional clean-target setup.
- This should be treated cautiously because the synthetic clean English target
  distribution is templated; the adapter may learn target priors rather than a
  broadly useful byte interface.

## Current Conclusion

Under the current Phase 2 harness, ByteBridge is not yet validated. The strongest
honest conclusion is:

> A fixed-patch byte adapter can be trained while keeping Qwen2.5-0.5B frozen,
> and it shows a typo-noise robustness signal, but it does not preserve clean
> performance and does not handle Unicode/multilingual/code/tokenizer-stress
> inputs better than the native tokenizer baseline.

## Recommended Next Steps

1. Add embedding distillation before LM-objective training so byte latents align
   with Qwen token embedding geometry.
2. Run latent length 64 only after distillation, because current l32 failure may
   be an information bottleneck.
3. Add multilingual/code/stress examples to the target-side training mixture only
   in a controlled run, then compare against clean/noise-only.
4. Add bootstrap confidence intervals after a run has plausible clean retention.
5. Keep the current results as a negative baseline; do not discard them.
