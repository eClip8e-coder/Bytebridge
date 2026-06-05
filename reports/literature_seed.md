# Literature Seed

This file records the initial primary-source map for ByteBridge.

## Byte/Character Models

- ByT5: byte-level T5; useful for token-free motivation, multilingual and noise evaluation.
  Source: <https://arxiv.org/abs/2105.13626>
- CANINE: character-level encoder with downsampling/upsampling; useful for compression design.
  Source: <https://arxiv.org/abs/2103.06874>
- Charformer: gradient-based subword tokenization; useful for learned pooling alternatives.
  Source: <https://arxiv.org/abs/2106.12672>
- MEGABYTE: multiscale byte modeling with byte patches; useful for patch-level latent framing.
  Source: <https://arxiv.org/abs/2305.07185>
- MambaByte: byte-level autoregressive Mamba; useful as a non-Transformer byte modeling reference.
  Source: <https://arxiv.org/abs/2401.13660>
- BLT: entropy-based byte patches; directly relevant to fixed vs dynamic patch ablations.
  Source: <https://arxiv.org/abs/2412.09871>

## Frozen LM Adaptation

- Prompt tuning and prefix tuning: freeze the LM and train continuous input-side or KV-side parameters.
  Sources: <https://aclanthology.org/2021.emnlp-main.243/>, <https://arxiv.org/abs/2101.00190>
- Adapter tuning and LoRA: parameter-efficient adaptation baselines, though they do not remove tokenizer input.
  Sources: <https://proceedings.mlr.press/v97/houlsby19a.html>, <https://arxiv.org/abs/2106.09685>
- Hugging Face causal LM `inputs_embeds`: the operational interface used by ByteBridge.
  Source: <https://huggingface.co/docs/transformers/en/model_doc/gpt2>

## Tokenizer Robustness and Bias

- Tokenization stress tests show brittleness under segmentation, typo, spacing, and Unicode variation.
  Sources: <https://arxiv.org/abs/2405.17067>, <https://arxiv.org/abs/2406.11687>
- Multilingual tokenization bias work motivates reporting per-language token/byte lengths, latency, and robustness drop.
  Sources: <https://arxiv.org/abs/2305.15425>, <https://arxiv.org/abs/2305.17179>
