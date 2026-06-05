# Phase 5 Soft Prefix Injection Report

Date: 2026-05-22

## Goal

Phase 5 tested whether input-layer-only ByteBridge was too weak. Instead of
trying to replace the tokenized prompt with byte latents, the adapter compresses
raw UTF-8 bytes into soft prefix embeddings:

```text
input_text bytes -> BytePrefixAdapter -> P soft prefix embeddings
target_text tokens -> frozen Qwen token embeddings
inputs_embeds = [byte prefix] + [target embeddings]
loss only on target tokens
```

The input side remains tokenizer-free. The target/output side still uses the
Qwen tokenizer, matching the prior paired conditional setup.

## Implemented Artifacts

New module:

- `bytebridge/adapters/prefix_adapter.py`

New scripts:

- `scripts/phase5_train_prefix.py`
- `scripts/phase5_compare.py`

Configs:

- `configs/phase5_prefix_p16.yaml`
- `configs/phase5_prefix_p32.yaml`
- `configs/phase5_prefix_p64.yaml`
- `configs/phase5_prefix_p32_smoke.yaml`

Outputs:

- `experiments/phase5/prefix_fixed_p16/`
- `experiments/phase5/prefix_fixed_p32/`
- `experiments/phase5/prefix_fixed_p64/`
- `experiments/phase5/tables/phase5_comparison.csv`
- `experiments/phase5/tables/phase5_summary.json`

KV-prefix was not implemented in this phase. The current report is Phase 5A:
soft prefix embeddings only.

## Main Results

Tokenizer baseline clean loss: 0.3089.

| run | clean loss | clean retention | winning buckets vs tokenizer | tokens/sec | peak alloc MiB |
|---|---:|---:|---|---:|---:|
| Phase 2 best | 0.6770 | 2.19 | typo light/medium/heavy | 24879 | 3467.6 |
| Phase 3 best | 0.6863 | 2.22 | typo light/medium/heavy | 24805 | 3467.6 |
| Phase 4 fixed recon | 0.5669 | 1.84 | typo light/medium/heavy | 12758 | 3466.1 |
| prefix_fixed_p16 | 0.6014 | 1.95 | typo light/medium/heavy | 28639 | 2973.7 |
| prefix_fixed_p32 | 0.6766 | 2.19 | typo light/medium/heavy | 24208 | 3463.8 |
| prefix_fixed_p64 | 0.5441 | 1.76 | typo light/medium/heavy | 17896 | 4447.8 |

Selected bucket losses:

| bucket | tokenizer | Phase 4 fixed recon | prefix p16 | prefix p32 | prefix p64 |
|---|---:|---:|---:|---:|---:|
| clean_english | 0.3089 | 0.5669 | 0.6014 | 0.6766 | 0.5441 |
| code | 0.3876 | 2.0058 | 0.9595 | 0.8682 | 0.7878 |
| tokenizer_stress | 0.4806 | 2.8533 | 1.5840 | 1.8324 | 1.5118 |
| unicode_stress | 0.5614 | 2.3900 | 1.2865 | 1.2473 | 0.9630 |
| typo_noise_light | 1.3838 | 0.6416 | 0.6486 | 0.6989 | 0.5817 |
| typo_noise_medium | 1.8428 | 0.6626 | 0.7151 | 0.7257 | 0.6097 |
| typo_noise_heavy | 2.8353 | 0.7749 | 0.7991 | 0.8167 | 0.6686 |

## Interpretation

Soft prefix injection helps stress buckets much more than input-layer prompt
replacement did:

- p64 improves code loss from Phase 4 fixed recon 2.0058 to 0.7878.
- p64 improves tokenizer-stress from 2.8533 to 1.5118.
- p64 improves Unicode from 2.3900 to 0.9630.
- typo-noise gains are preserved.

However, clean retention is still above the promising threshold:

- required for promising: <= 1.5
- required for strong pass: <= 1.2
- best observed soft prefix: 1.76

The tokenizer baseline still dominates all non-typo buckets in absolute loss.

## Prefix Length Trade-off

p64 is best on clean and stress buckets, but costs more:

- p16: 28.6k target tokens/sec, 2.97 GiB peak allocated
- p32: 24.2k target tokens/sec, 3.46 GiB peak allocated
- p64: 17.9k target tokens/sec, 4.45 GiB peak allocated

Longer prefix helps, but the improvement is not enough to satisfy the success
criteria.

## Pass/Fail

Passed:

- LLM trainable params remained 0.
- checksum delta remained 0.0.
- all results are reported by bucket.
- Phase 2 test manifest was reused unchanged.
- soft prefix substantially improves Unicode/code/tokenizer-stress compared with
  prior ByteBridge variants.

Failed:

- no prefix length reached clean retention <= 1.5.
- no prefix run beat tokenizer baseline outside typo-noise buckets.
- p64 improvement comes with lower throughput and higher memory.
- p32 was not promising, so extra seeds were not run.

## Conclusion

Phase 5A is a diagnostic partial result, not a success.

Soft prefix injection is clearly more useful than pure input-layer replacement
for code, Unicode, and tokenizer-stress buckets. This supports the hypothesis
that input-layer-only ByteBridge was too weak.

But soft prefix embeddings still do not make frozen Qwen competitive with its
native tokenizer baseline. The best clean retention is 1.76, short of the 1.5
promising threshold and far from the 1.2 pass threshold.

## Next Step

The next diagnostic should be KV-prefix injection if the installed Qwen2
Transformers API allows stable `past_key_values` construction. If KV-prefix also
fails, the project should pivot to a strong negative result: frozen Qwen2.5-0.5B
does not easily accept a tokenizer-free byte input interface through small
external adapters under this setup.
