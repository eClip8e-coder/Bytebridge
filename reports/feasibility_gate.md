# Feasibility Gate Report

Date: 2026-05-22

## Question

Can a learnable UTF-8 byte adapter feed a frozen causal LM through
`inputs_embeds`, receive gradients, and reduce a small clean-text training loss
without updating the LM?

## Setup

- Model: `Qwen/Qwen2.5-0.5B`
- GPU: NVIDIA GeForce RTX 5090, 32 GB
- Python: 3.12.3
- PyTorch: 2.12.0+cu130
- Transformers: 5.9.0
- Adapter: byte embedding -> fixed 4-byte patches -> MLP local encoder -> projection
- Max bytes: 128
- Latents: 32
- Batch size: 4
- Steps: 120
- Trainable LLM parameters: 0
- Adapter parameters: 922,240

## Result

The Feasibility Gate passed.

| Metric | Value |
|---|---:|
| Initial loss | 12.6508 |
| Final loss | 4.3031 |
| Loss decrease | 8.3477 |
| Max allocated GPU memory | 1473.7 MiB in training log |
| Max reserved GPU memory | 1576.0 MiB in training log |
| Total training time | 2.92 sec |
| Adapter gradient seen | yes |
| LLM gradient seen | no |
| LLM checksum delta, first 4 tensors | 0.0 |

Artifacts:

- Metrics: `experiments/feasibility/qwen05/metrics.csv`
- Environment: `experiments/feasibility/qwen05/env.json`
- Summary: `experiments/feasibility/qwen05/summary.json`
- Checkpoint: `experiments/feasibility/qwen05/adapter.pt`
- Plot: `experiments/feasibility/qwen05/loss_curve.png`

## Interpretation

This result only establishes interface feasibility: a byte-to-latent adapter can
drive a frozen Qwen2.5-0.5B through `inputs_embeds`, gradients reach only the
adapter, and a tiny clean-text objective can be optimized.

It does not yet show tokenizer robustness, multilingual gains, or meaningful
language modeling quality. The current label alignment is intentionally minimal:
fixed byte patches produce 32 latents and token labels are truncated/padded to
that length. The next phase must implement stronger baselines and alignment
objectives before making any robustness claim.

## Notes

The system lacked `python3-venv`, `pip`, `conda`, and sudo access. User-level
`pip` was bootstrapped with `get-pip.py --user --break-system-packages`, and
dependencies were installed into the user Python environment. The exact user
package list is saved at `experiments/feasibility/requirements_user.txt`.
