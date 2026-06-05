# Phase 6 KV-Prefix Injection Report

Date: 2026-05-22

## Goal

Phase 6 tested the last major diagnostic route for ByteBridge: inject
byte-derived information directly into the frozen Qwen attention cache instead
of only replacing input embeddings or adding soft prefix embeddings.

The core question was:

```text
Can byte-derived KV-prefix memory let frozen Qwen/Qwen2.5-0.5B use tokenizer-free
input information more effectively than soft prefix embeddings?
```

The model backbone and LM head remained frozen. The Phase 2 train/validation/test
manifests were reused unchanged.

## KV Cache Probe

Script:

- `scripts/phase6_probe_qwen_kv_cache.py`

Output:

- `experiments/phase6/kv_probe/summary.json`
- `experiments/phase6/kv_probe/env.json`

Observed Qwen2.5-0.5B cache configuration:

| field | value |
|---|---:|
| hidden size | 896 |
| layers | 24 |
| attention heads | 14 |
| key/value heads | 2 |
| head dim | 64 |
| cache type | `DynamicCache` |
| layer K/V shape | `[batch, 2, seq, 64]` |

Manual `DynamicCache` construction worked. Legacy tuple-style
`past_key_values` did not work in this installed Transformers version:
`AttributeError: 'tuple' object has no attribute 'get_seq_length'`.

## Method

Implemented module:

- `bytebridge/adapters/kv_prefix_adapter.py`

Training script:

- `scripts/phase6_train_kv_prefix.py`

Comparison script:

- `scripts/phase6_compare.py`

Configs:

- `configs/phase6_kv_prefix_smoke_p8_l4.yaml`
- `configs/phase6_kv_prefix_p16_l4.yaml`
- `configs/phase6_kv_prefix_p32_l4.yaml`
- `configs/phase6_kv_prefix_p32_lall.yaml`

The adapter maps raw UTF-8 bytes to per-layer K/V tensors:

```text
input_text bytes
-> byte embedding
-> local byte encoder
-> fixed byte pooling
-> prefix hidden states
-> per-layer K/V tensors
-> Qwen DynamicCache
```

The conditional LM setup was:

```text
input_text -> byte KV prefix
target_text -> Qwen token ids / token embeddings
loss only on target tokens
```

For target-side teacher forcing, the script prepends one EOS/start token and
supervises the following target tokens. The target/output side still uses the
Qwen tokenizer, matching prior paired conditional evaluations.

For `first_n` runs, the injected layers receive learned byte-derived K/V tensors.
Non-injected layers receive zero K/V tensors with the same prefix length so that
the cache sequence length stays consistent across Qwen layers. This is a
diagnostic engineering compromise, not a claim that zero-filled non-injected
layers are an optimal prefix-tuning implementation.

## Runs

| run | prefix | layers | steps | adapter params | LLM trainable | checksum delta |
|---|---:|---:|---:|---:|---:|---:|
| `kv_prefix_smoke_p8_l4` | 8 | first 4 | 20 | 859,904 | 0 | 0.0 |
| `kv_prefix_p16_l4` | 16 | first 4 | 2000 | 860,928 | 0 | 0.0 |
| `kv_prefix_p32_l4` | 32 | first 4 | 2000 | 865,024 | 0 | 0.0 |
| `kv_prefix_p32_lall` | 32 | all 24 | 2000 | 2,180,864 | 0 | 0.0 |

The smoke run verified forward/backward, nonzero adapter gradients, zero LLM
gradients, and checksum stability.

## Main Results

Tokenizer baseline clean loss: 0.3089.

| run | clean loss | clean retention | winning buckets vs tokenizer | tokens/sec | peak reserved MiB |
|---|---:|---:|---|---:|---:|
| Phase 2 best | 0.6770 | 2.19 | typo light/medium/heavy | 24879 | 7868 |
| Phase 4 fixed recon | 0.5669 | 1.84 | typo light/medium/heavy | 12758 | 7592 |
| Phase 5 soft prefix p64 | 0.5441 | 1.76 | typo light/medium/heavy | 17896 | 10538 |
| KV p16 first 4 | 5.5084 | 17.83 | none | 31193 | 4094 |
| KV p32 first 4 | 1.7407 | 5.64 | typo heavy | 33250 | 3982 |
| KV p32 all layers | 0.4985 | 1.61 | typo light/medium/heavy | 33235 | 4000 |

Selected bucket losses:

| bucket | tokenizer | Phase 4 fixed recon | soft prefix p64 | KV p16 l4 | KV p32 l4 | KV p32 all |
|---|---:|---:|---:|---:|---:|---:|
| clean_english | 0.3089 | 0.5669 | 0.5441 | 5.5084 | 1.7407 | 0.4985 |
| code | 0.3876 | 2.0058 | 0.7878 | 7.1809 | 2.8560 | 1.3462 |
| tokenizer_stress | 0.4806 | 2.8533 | 1.5118 | 6.8827 | 3.8917 | 1.5969 |
| unicode_stress | 0.5614 | 2.3900 | 0.9630 | 8.3641 | 4.2160 | 0.9809 |
| typo_noise_light | 1.3838 | 0.6416 | 0.5817 | 5.3859 | 1.8655 | 0.5167 |
| typo_noise_medium | 1.8428 | 0.6626 | 0.6097 | 5.4577 | 1.8964 | 0.5298 |
| typo_noise_heavy | 2.8353 | 0.7749 | 0.6686 | 5.5110 | 2.0487 | 0.5986 |
| multilingual_zh | 0.6085 | 4.7134 | 4.9432 | 8.4665 | 6.0549 | 11.1217 |
| multilingual_ar | 0.3806 | 4.4678 | 5.4298 | 11.8669 | 7.2254 | 10.7503 |

Full per-bucket comparison:

- `experiments/phase6/tables/phase6_comparison.csv`
- `experiments/phase6/tables/phase6_summary.json`

## Interpretation

The first-4-layer KV prefix variants were not useful. `kv_prefix_p16_l4`
collapsed badly, and `kv_prefix_p32_l4` remained far worse than the Phase 5 soft
prefix run.

The all-layer KV-prefix run is the important diagnostic result:

- It improved clean retention from soft prefix p64 `1.76x` to `1.61x`.
- It preserved typo-noise gains.
- It improved over soft prefix p64 on clean and typo buckets.
- It did not improve code/tokenizer-stress/Unicode over soft prefix p64.
- It badly failed multilingual non-Latin buckets.
- It still did not beat tokenizer baseline outside typo-noise buckets.

This supports a narrow version of the deeper-injection hypothesis: injecting
byte-derived memory into every layer helps more than shallow input replacement
or first-layer-only context. However, the improvement is not enough to make
ByteBridge competitive with native tokenization.

## Success Criteria

Promising threshold:

- checksum delta = 0: passed
- LLM trainable params = 0: passed
- clean retention <= 1.5: failed, best was 1.61
- better than soft prefix p64: partially passed on clean, failed on several
  stress buckets
- at least two non-typo buckets improved meaningfully: failed versus soft prefix
  p64
- typo gains preserved: passed for all-layer KV

Strong pass threshold:

- clean retention <= 1.2: failed
- multiple noisy/stress buckets beat tokenizer baseline: failed
- multi-seed consistency: not run because single-seed result did not reach the
  promising threshold

## Conclusion

Phase 6 is a diagnostic partial result, but it does not pass.

True Qwen `DynamicCache` KV-prefix injection is feasible in this environment, and
all-layer KV injection is the best deeper-injection result so far. It improves
clean retention from `1.76x` to `1.61x` relative to the Phase 5 soft prefix
best, while keeping the LLM frozen and checksum-stable.

But it still misses the `<=1.5x` promising threshold, remains far from the
`<=1.2x` pass threshold, and does not beat the tokenizer baseline on any
non-typo bucket. Multilingual buckets remain especially poor.

The current evidence supports a strong negative conclusion for the tested setup:
under a single RTX 5090, frozen `Qwen/Qwen2.5-0.5B`, small external byte
adapters, and unchanged Phase 2 evaluation, tokenizer-free byte input adapters
do not become a practical replacement for the native tokenizer. Deeper injection
helps, but not enough.

## Recommended Next Step

Stop adding more structural variants unless the research goal changes. The next
useful deliverable is a final technical report that preserves the full negative
trajectory:

1. input-layer ByteBridge is feasible but weak;
2. embedding distillation reduces alignment loss but does not improve LM eval;
3. token reconstruction and oracle boundaries diagnose a boundary/token-identity
   bottleneck but do not fix clean retention;
4. soft prefix and all-layer KV-prefix show deeper injection helps;
5. frozen Qwen still strongly prefers its native tokenizer across non-typo
   stress settings.
