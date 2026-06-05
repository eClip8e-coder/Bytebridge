# Phase 4 Boundary Reconstruction Report

Date: 2026-05-22

## Goal

Phase 4 tested whether ByteBridge fails because fixed byte patches do not
recover token boundary or token identity structure. This phase added
token-byte mapping, boundary-aware span pooling, token reconstruction
pretraining, and controlled frozen-LM evaluation.

The Phase 2 test manifest and tokenizer baseline were reused unchanged.

## Implemented Artifacts

New modules:

- `bytebridge/data/token_byte_mapping.py`
- `bytebridge/adapters/boundary_adapter.py`

New scripts:

- `scripts/phase4_mapping_stats.py`
- `scripts/phase4_train_token_reconstruction.py`
- `scripts/phase4_train_after_reconstruction.py`
- `scripts/phase4_compare.py`

New outputs:

- `experiments/phase4/mapping_stats/`
- `experiments/phase4/recon_fixed_l32/`
- `experiments/phase4/recon_oracle_boundary_l32/`
- `experiments/phase4/recon_heuristic_boundary_l32/`
- `experiments/phase4/bb_recon_fixed_l32_then_lm/`
- `experiments/phase4/bb_recon_oracle_boundary_l32_then_lm/`
- `experiments/phase4/bb_recon_heuristic_boundary_l32_then_lm/`
- `experiments/phase4/tables/phase4_comparison.csv`
- `experiments/phase4/tables/phase4_summary.json`

## Token-Byte Mapping

Qwen tokenizer offset mapping worked cleanly on the Phase 2 test set:

- unmapped token rate: 0.0 for all buckets
- mapping examples: `experiments/phase4/mapping_stats/mapping_examples.jsonl`
- summary: `experiments/phase4/mapping_stats/mapping_summary.json`

Oracle boundary uses tokenizer-derived token byte spans. It is a diagnostic
upper bound and is not tokenizer-free.

## Reconstruction Results

Token reconstruction used a trainable classifier head during pretraining. The
head is not used during LM evaluation; only the adapter is transferred.

| run | boundary | final train loss | final top1 | final top5 | checksum |
|---|---|---:|---:|---:|---:|
| recon_fixed_l32 | fixed 4-byte patches | 1.6455 | 0.6119 | 0.8598 | 0.0 |
| recon_oracle_boundary_l32 | tokenizer oracle spans | 0.3909 | 0.8983 | 0.9831 | 0.0 |
| recon_heuristic_boundary_l32 | UTF-8/script/punct heuristic | 3.1292 | 0.4297 | 0.5729 | 0.0 |

Validation highlights:

| run | clean top1 | code top1 | tokenizer-stress top1 | unicode top1 |
|---|---:|---:|---:|---:|
| fixed | 0.8669 | 0.8351 | 0.8482 | 0.8581 |
| oracle | 0.9945 | 0.9632 | 0.9513 | 0.9254 |
| heuristic | 0.5509 | 0.4862 | 0.6030 | 0.6506 |

Interpretation: token identity reconstruction is very learnable, and oracle
token boundaries are much better than fixed or heuristic boundaries for this
objective. This supports the hypothesis that boundary/token identity structure
is a real bottleneck.

## Downstream LM Results

Tokenizer baseline clean loss: 0.3089.

| run | clean loss | clean retention | winning buckets vs tokenizer |
|---|---:|---:|---|
| Phase 2 best: bb_noise_l32_seed1_2k | 0.6770 | 2.19 | typo light/medium/heavy |
| Phase 3 best: bb_distill_then_lm_l32 | 0.6863 | 2.22 | typo light/medium/heavy |
| bb_recon_fixed_l32_then_lm | 0.5669 | 1.84 | typo light/medium/heavy |
| bb_recon_oracle_boundary_l32_then_lm | 0.6794 | 2.20 | typo light/medium/heavy |
| bb_recon_heuristic_boundary_l32_then_lm | 0.6188 | 2.00 | typo light/medium/heavy |

Selected bucket losses:

| bucket | tokenizer | Phase 2 best | fixed recon | oracle recon | heuristic recon |
|---|---:|---:|---:|---:|---:|
| clean_english | 0.3089 | 0.6770 | 0.5669 | 0.6794 | 0.6188 |
| typo_noise_light | 1.3838 | 0.7454 | 0.6416 | 0.7947 | 0.6623 |
| typo_noise_medium | 1.8428 | 0.7786 | 0.6626 | 0.8547 | 0.6830 |
| typo_noise_heavy | 2.8353 | 0.8203 | 0.7749 | 0.9766 | 0.7385 |
| unicode_stress | 0.5614 | 3.8226 | 2.3900 | 1.7040 | 2.8423 |
| code | 0.3876 | 2.4261 | 2.0058 | 1.5469 | 1.9823 |
| tokenizer_stress | 0.4806 | 3.1706 | 2.8533 | 2.5527 | 2.9037 |
| multilingual_zh | 0.6085 | 4.5419 | 4.7134 | 4.4050 | 4.3265 |
| multilingual_ja | 0.5565 | 4.1580 | 3.7556 | 3.7328 | 3.7665 |

## Answers to Phase 4 Questions

### Is fixed patch failure caused by boundary errors?

Partly. Oracle token boundaries dramatically improve token reconstruction, and
oracle-boundary LM improves Unicode/code/tokenizer-stress losses relative to
Phase 2. Boundary quality matters.

But boundary is not the whole problem: oracle-boundary LM does not improve clean
retention and does not beat tokenizer baseline outside typo buckets.

### Does oracle token boundary significantly improve downstream LM?

It improves stress buckets but not clean retention:

- Unicode: 3.8226 -> 1.7040 vs Phase 2 best
- Code: 2.4261 -> 1.5469
- Tokenizer-stress: 3.1706 -> 2.5527
- Clean: 0.6770 -> 0.6794, effectively no improvement

This suggests oracle boundary helps represent difficult strings, but frozen Qwen
still does not use these input-layer byte latents as well as native token
embeddings.

### Does heuristic tokenizer-free boundary approach oracle?

No. Heuristic reconstruction is much worse than oracle reconstruction. In LM
evaluation, heuristic improves over Phase 2 in clean/code/stress somewhat, but
does not approach tokenizer baseline or oracle reconstruction quality.

### Is token reconstruction more useful than embedding distillation?

Yes, in this implementation. Fixed reconstruction pretraining improves clean
retention from 2.19 to 1.84 and improves several stress buckets. Projection
alignment in Phase 3 did not improve clean retention.

### If reconstruction works but LM still fails, what does that imply?

It suggests that recovering token identity in a classifier head is not enough.
The frozen LM likely needs the internal structure induced by tokenizer inputs,
including token boundaries, positional semantics, and possibly deeper-layer
activation patterns. Input-layer latents alone remain too weak.

### Should the project turn toward prefix/KV injection?

Yes. Phase 4 gives enough evidence to justify Phase 4B: keep the LLM frozen, but
inject byte-derived information as soft prefix tokens or shallow KV prefixes
rather than only replacing input embeddings.

## Pass/Fail

Passed:

- LLM trainable parameters remained 0.
- checksum delta remained 0.0.
- token-byte mapping had 0 unmapped tokens on the test suite.
- token reconstruction learned strong signal, especially oracle-boundary.
- fixed reconstruction improved clean retention over Phase 2/3.
- Unicode/code/tokenizer-stress improved under oracle-boundary LM.

Failed:

- clean retention did not reach <= 1.5, let alone <= 1.2.
- no Phase 4 run beat tokenizer baseline beyond typo-noise buckets.
- heuristic tokenizer-free boundary did not approach oracle reconstruction.
- oracle-boundary itself did not fix clean retention.

## Current Conclusion

Phase 4 is a diagnostic partial result, not a success.

Token reconstruction is more useful than embedding MSE, and boundary quality is
important. However, even oracle token boundaries plus token reconstruction do
not make input-layer ByteBridge competitive with the native tokenizer baseline.

The next scientifically justified step is not more seeds on input-only
ByteBridge. The next diagnostic should test prefix/KV injection with the LLM
still frozen.
