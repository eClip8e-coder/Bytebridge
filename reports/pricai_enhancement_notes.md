# PRICAI 2026 Enhancement Notes

Date: 2026-05-27

This note records the first enhancement batch for a PRICAI 2026 main-conference version of ByteBridge. The goal is to address the most likely reviewer objections to the TokShop workshop draft:

1. the typo-noise gains may be explainable by simple preprocessing;
2. the failure may be specific to Qwen2.5-0.5B;
3. the tokenizer-stress story needs more mechanism-oriented analysis.

## 1. Tokenizer Preprocessing Controls

Script:

- `scripts/pricai_eval_tokenizer_transforms.py`

Outputs:

- `experiments/pricai/tokenizer_transforms/*`
- `experiments/pricai/tables/tokenizer_transform_comparison.csv`
- `experiments/pricai/plots/preprocessing_typo_controls.png`

All controls keep the same target text and compute loss only on target tokens. Transformations are applied only to the input side.

| mode | changed frac | clean | typo light | typo medium | typo heavy | unicode |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.000 | 0.3089 | 1.3838 | 1.8428 | 2.8353 | 0.5614 |
| nfc | 0.015 | 0.3089 | 1.3838 | 1.8428 | 2.8353 | 0.5614 |
| nfkc | 0.052 | 0.3089 | 1.3838 | 1.8428 | 2.8353 | 0.8658 |
| whitespace | 0.112 | 0.3089 | 1.3518 | 1.8257 | 2.8137 | 0.5614 |
| punctfold | 0.023 | 0.3089 | 1.3838 | 1.8428 | 2.8353 | 0.6172 |
| casefold | 0.693 | 0.6028 | 1.0721 | 1.4980 | 2.4380 | 0.5840 |
| repeat_squeeze | 0.039 | 0.3089 | 1.3835 | 1.8429 | 2.8189 | 0.5614 |
| typo_cleanup | 0.709 | 0.6028 | 1.0748 | 1.5098 | 2.4155 | 0.6399 |
| nfkc_cleanup | 0.261 | 0.3603 | 1.4330 | 1.9065 | 2.8558 | 1.0500 |
| oracle_clean_typo | 0.296 | 0.3089 | 0.3039 | 0.3039 | 0.3039 | 0.5614 |

Interpretation:

- Conservative preprocessing (`NFC`, `whitespace`, `repeat_squeeze`) does not explain away ByteBridge typo-noise wins.
- Stronger typo cleanup (`casefold`, `typo_cleanup`) improves typo buckets but damages clean loss substantially.
- ByteBridge Phase 5/6 typo losses remain better than deterministic cleanup:
  - Phase 5 p64: 0.5817 / 0.6097 / 0.6686
  - Phase 6 KV all: 0.5167 / 0.5298 / 0.5986
  - `typo_cleanup`: 1.0748 / 1.5098 / 2.4155
- `oracle_clean_typo` is only an upper bound; it uses the clean target as the prompt for typo buckets and must not be reported as a deployable baseline.

PRICAI paper use:

- State that ByteBridge beats raw tokenizer and deterministic cleanup on typo-noise buckets.
- Also state that the typo result is still narrow: it does not imply broad tokenizer-free superiority.

## 2. Fragmentation Analysis

Script:

- `scripts/pricai_fragmentation_analysis.py`

Outputs:

- `experiments/pricai/fragmentation/sample_fragmentation.csv`
- `experiments/pricai/fragmentation/bucket_fragmentation.csv`
- `experiments/pricai/fragmentation/fragmentation_summary.json`
- `experiments/pricai/plots/bucket_loss_and_fragmentation.png`

Runs compared:

- tokenizer baseline
- Phase 5 soft prefix p64
- Phase 6 KV p32 all layers

Selected correlations over all test samples:

| predictor | outcome | Pearson | Spearman |
|---|---|---:|---:|
| input token length | tokenizer loss | 0.669 | 0.214 |
| tokens per byte | tokenizer loss | 0.227 | 0.334 |
| bytes per token | tokenizer loss | -0.319 | -0.334 |
| input token length | Phase 5 gain vs tokenizer | 0.276 | 0.459 |
| input token length | Phase 6 gain vs tokenizer | 0.349 | 0.477 |

Interpretation:

- Longer tokenized inputs correlate with higher tokenizer loss, but fragmentation alone does not explain all outcomes.
- Byte adapters gain most clearly on typo buckets, not on all high-fragmentation buckets.
- Multilingual failures are not solved by byte input; this suggests the bottleneck is not only tokenizer fragmentation, but also compression, script coverage in adapter training, and mismatch with frozen-LM token semantics.

PRICAI paper use:

- Use fragmentation analysis as mechanism evidence, not causal proof.
- Present one bucket-level table or plot plus a concise correlation table.

## 3. Second-Model Minimal Replication: Qwen2.5-1.5B

Reason for model choice:

- Same Qwen2.5 family, minimizing engineering confounds.
- Same tokenizer family and compatible Hugging Face/Qwen cache API.
- Tests whether the negative result is unique to Qwen2.5-0.5B or persists with a larger frozen Qwen model.

Runs:

- tokenizer baseline
- soft prefix p64, 2k steps
- soft prefix p64, 8k steps
- all-layer KV prefix p32, 2k steps

Outputs:

- `experiments/pricai/qwen15/tokenizer_baseline/`
- `experiments/pricai/qwen15/prefix_fixed_p64/`
- `experiments/pricai/qwen15/prefix_fixed_p64_8k/`
- `experiments/pricai/qwen15/kv_prefix_p32_lall/`
- `experiments/pricai/tables/qwen15_second_model_comparison.csv`
- `experiments/pricai/tables/qwen15_second_model_summary.json`

Tokenizer clean baseline:

- Qwen2.5-1.5B clean loss: 0.3274

| run | clean loss | clean retention | wins vs tokenizer |
|---|---:|---:|---|
| qwen15_prefix_p64_2k | 2.0351 | 6.22x | none |
| qwen15_prefix_p64_8k | 0.8927 | 2.73x | typo heavy |
| qwen15_kv_p32_all | 0.8245 | 2.52x | typo medium, typo heavy |

All runs:

- LLM trainable params: 0
- checksum delta: 0.0

Interpretation:

- The second model does not make ByteBridge competitive.
- Longer p64 training improves clean retention but remains far above the 1.5x promising threshold.
- KV-prefix is the best Qwen1.5B variant tested, but still has 2.52x clean retention and beats tokenizer baseline only on typo medium/heavy.
- This strengthens the PRICAI claim that the failure is not just a Qwen2.5-0.5B artifact, while still keeping the claim scoped to the Qwen2.5 family.

## 4. Recommended Next Work Before PRICAI Submission

High priority:

1. Integrate preprocessing control table into the paper.
2. Integrate Qwen2.5-1.5B second-model table.
3. Add fragmentation analysis figure/table and a careful paragraph.
4. Rewrite the TokShop paper into a PRICAI archival main-paper format.

Medium priority:

1. Run a small fixed-input adapter on Qwen2.5-1.5B only if space/time permits.
2. Add one real/curated stress slice only if it can be done without changing the core benchmark or weakening reproducibility.

Avoid before PRICAI:

1. Full Phase 1-6 replication on Qwen1.5B.
2. Large hyperparameter sweeps.
3. External spellchecker/LLM cleanup baselines.
4. Partial LM finetuning, because it changes the frozen-LM thesis.

