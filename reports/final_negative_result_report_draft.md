# ByteBridge Final Technical Report Draft

Working title:

**ByteBridge: Negative Evidence for Tokenizer-Free Byte Input Adapters on a Frozen
Qwen2.5-0.5B Language Model**

Date: 2026-05-22

## Abstract

ByteBridge studies whether a learnable tokenizer-free byte input adapter can
partially replace the fixed tokenizer of a frozen causal language model. The
project keeps `Qwen/Qwen2.5-0.5B`, its LM head, and the tokenizer-side output
space frozen, and trains only small byte-derived adapters. Across six phases, we
tested input-layer byte latents, projection alignment, token reconstruction,
boundary-aware patching, soft prefix injection, and KV-prefix injection. The
main result is negative: byte adapters consistently improve typo-noise
robustness, but they do not approach the native tokenizer baseline on clean
English, multilingual text, Unicode-heavy strings, code, or tokenizer-stress
inputs. The best deeper-injection result, all-layer KV-prefix, reaches `1.61x`
clean retention relative to the tokenizer baseline, missing the `<=1.5x`
promising threshold and the `<=1.2x` pass threshold. Under the tested single-GPU,
frozen-Qwen, small-adapter setup, ByteBridge is not a practical replacement for
native tokenization.

## 1. Introduction

Fixed subword tokenizers are brittle under spelling noise, rare Unicode,
multilingual scripts, and code-like strings. Byte-level models avoid tokenizer
coverage issues but usually require training or retraining the language model
itself. ByteBridge asks a narrower question: can a small byte-to-latent adapter
be trained while the language model remains frozen?

This is attractive because it would preserve an existing LLM while improving
input robustness. It is also difficult because frozen decoder-only LMs are
trained around tokenizer-induced embedding, position, and attention semantics.

## 2. Research Question

Can a tokenizer-free byte input adapter provide useful prompt/context
representations for a frozen causal LM, while preserving clean text performance
and improving robustness on tokenizer-unfriendly inputs?

Constraints:

- frozen `Qwen/Qwen2.5-0.5B` backbone;
- frozen LM head;
- train only byte adapter parameters;
- input side tokenizer-free;
- target/output side may use the original Qwen tokenizer;
- single RTX 5090 32GB reproducibility target;
- Phase 2 train/validation/test manifests fixed after construction.

## 3. Method Summary

The project tested increasingly structured adapters:

1. Fixed-patch input-layer ByteAdapter via `inputs_embeds`.
2. Embedding distillation to align byte latents with Qwen token embeddings.
3. Token-byte mapping, boundary-aware patching, and token reconstruction.
4. Soft prefix embeddings that condition target tokens without replacing them.
5. KV-prefix injection through Qwen `DynamicCache`.

All conditional evaluations use:

```text
input_text as context
target_text as supervised suffix
loss computed only on target tokens
```

For noisy examples, `input_text` is noisy and `target_text` is clean.

## 4. Experimental Setup

Model:

- `Qwen/Qwen2.5-0.5B`

Evaluation buckets:

- clean English;
- typo noise light/medium/heavy;
- Unicode stress;
- multilingual text;
- code;
- tokenizer-stress strings.

Primary metrics:

- target-token LM loss;
- clean retention: `run_clean_loss / tokenizer_clean_loss`;
- per-bucket wins versus tokenizer baseline;
- throughput and GPU memory;
- adapter parameter count;
- LLM trainable parameter count;
- frozen-model checksum delta.

Tokenizer baseline clean loss:

- `0.3089`

## 5. Results

| stage | best run | clean loss | clean retention | wins vs tokenizer baseline |
|---|---|---:|---:|---|
| Phase 2 | fixed input-layer noise l32 | 0.6770 | 2.19 | typo light/medium/heavy |
| Phase 3 | distill then LM l32 | 0.6863 | 2.22 | typo light/medium/heavy |
| Phase 4 | fixed recon then LM | 0.5669 | 1.84 | typo light/medium/heavy |
| Phase 5 | soft prefix p64 | 0.5441 | 1.76 | typo light/medium/heavy |
| Phase 6 | KV prefix p32 all layers | 0.4985 | 1.61 | typo light/medium/heavy |

Selected bucket losses:

| bucket | tokenizer | Phase 2 best | Phase 4 fixed recon | Phase 5 prefix p64 | Phase 6 KV all |
|---|---:|---:|---:|---:|---:|
| clean_english | 0.3089 | 0.6770 | 0.5669 | 0.5441 | 0.4985 |
| code | 0.3876 | 2.4261 | 2.0058 | 0.7878 | 1.3462 |
| tokenizer_stress | 0.4806 | 3.1706 | 2.8533 | 1.5118 | 1.5969 |
| unicode_stress | 0.5614 | 3.8226 | 2.3900 | 0.9630 | 0.9809 |
| typo_noise_light | 1.3838 | 0.7454 | 0.6416 | 0.5817 | 0.5167 |
| typo_noise_medium | 1.8428 | 0.7786 | 0.6626 | 0.6097 | 0.5298 |
| typo_noise_heavy | 2.8353 | 0.8203 | 0.7749 | 0.6686 | 0.5986 |

The consistent positive signal is typo-noise robustness. The consistent failure
is clean retention and non-typo stress performance relative to native
tokenization.

## 6. Phase-by-Phase Findings

### Phase 1: Feasibility Gate

The basic engineering path works. Qwen accepts `inputs_embeds`, adapter gradients
are nonzero, LLM gradients remain zero, and toy training loss decreases.

### Phase 2: Fixed Input-Layer ByteAdapter

Fixed byte patches are insufficient. They improve typo-noise buckets but produce
poor clean retention and weak Unicode/code/tokenizer-stress results.

### Phase 3: Projection Alignment

Embedding distillation loss drops strongly, but downstream LM evaluation does not
improve. Matching the embedding manifold with simple MSE/cosine objectives is
not enough.

### Phase 4: Boundary Reconstruction

Token reconstruction is more diagnostic than embedding distillation. Oracle token
boundaries greatly improve reconstruction accuracy, confirming a boundary/token
identity bottleneck. However, even reconstruction-pretrained adapters do not
repair downstream clean retention.

### Phase 5: Soft Prefix Injection

Soft prefix conditioning helps code, Unicode, and tokenizer-stress buckets much
more than input-layer replacement. This indicates input-layer-only adapters are
too weak. The best soft prefix still has clean retention `1.76x`, which misses
the promising threshold.

### Phase 6: KV-Prefix Injection

Qwen `DynamicCache` manual KV-prefix injection is feasible. All-layer KV-prefix
is the best deeper-injection result, with clean retention `1.61x`, but still
does not reach `<=1.5x` and does not beat tokenizer baseline outside typo-noise.

## 7. Interpretation

The experiments suggest four bottlenecks:

1. Fixed byte patching destroys tokenizer-relevant boundary structure.
2. Simple embedding alignment does not recover token identity.
3. Frozen Qwen behavior is strongly tied to native token embeddings and
   tokenizer-induced positional/semantic structure.
4. Deeper injection helps, but small byte adapters still cannot provide a
   competitive replacement for native tokenized prompts.

The project therefore supports a negative but useful claim: tokenizer-free input
interfaces for frozen LMs are not automatically obtained by training a small
byte adapter, even when the adapter is connected through increasingly powerful
interfaces.

## 8. Limitations

This result is scoped to:

- `Qwen/Qwen2.5-0.5B`;
- small adapters;
- single RTX 5090 experiments;
- synthetic/curated Phase 2 benchmark;
- output-side tokenizer still present;
- short training runs around 1k-2k steps for most diagnostics.

It does not rule out:

- larger byte adapters;
- longer training;
- joint tokenizer/LM adaptation;
- architectures trained from scratch for bytes;
- stronger contrastive or retrieval-style pretraining;
- deeper trainable side networks with more capacity.

## 9. Conclusion

ByteBridge produced a clear engineering feasibility result and several useful
diagnostics, but not a successful tokenizer replacement.

The strongest conclusion is negative:

```text
Under frozen Qwen2.5-0.5B, single-GPU, small-adapter constraints, tokenizer-free
byte input adapters do not match the native tokenizer baseline. They reliably
help typo-noise robustness, but fail clean retention and non-typo stress
benchmarks.
```

The next publishable direction is to frame ByteBridge as a negative result and
diagnostic study of why frozen tokenizer-based LMs resist byte-level input
replacement.

## 10. Reproducibility Pointers

Key reports:

- `reports/feasibility_gate.md`
- `reports/phase2_research_report.md`
- `reports/phase3_projection_alignment_report.md`
- `reports/phase4_boundary_reconstruction_report.md`
- `reports/phase5_prefix_kv_injection_report.md`
- `reports/phase6_kv_prefix_report.md`

Key comparison outputs:

- `experiments/phase6/tables/phase6_comparison.csv`
- `experiments/phase6/tables/phase6_summary.json`

The final reported experiments keep:

- LLM trainable params = 0;
- LM head frozen;
- checksum delta = 0.0 for adapter-trained runs;
- Phase 2 test manifest unchanged.
