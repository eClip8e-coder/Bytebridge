# PRICAI 2026 Draft Checklist

Date: 2026-05-28

## Files

- Local PDF: `D:\Desktop\dailyusage\bytebridge_pricai2026\main.pdf`
- Local source zip: `D:\Desktop\dailyusage\bytebridge_pricai2026\pricai2026_source.zip`
- Remote paper folder: `/home/normal/bytebridge/paper/pricai2026/`

## Format

- Template: Springer LNCS/LNAI (`llncs.cls`, `splncs04.bst`)
- Compiled with remote Tectonic: passed
- Updated cross-family draft page count: 16 pages including references
- PRICAI limit checked from CFP: 16 pages including references
- Remaining warnings: minor underfull boxes only
- Final polishing pass: Qwen diagnostic ladder compressed; cross-model mixed-result framing kept consistent
- Bibliography expanded to 23 verified real references; no invented citations added
- Abstract revised to follow problem -> gap -> method ladder -> mixed result -> boundary structure
- Springer formatting pass: multi-citation groups sorted numerically, headings converted to title case, and the remaining figure explicitly cross-referenced in the text
- Cross-model diagnosis added: KV-cache geometry comparison and multilingual failure case analysis; preprocessing figure removed to keep the paper within the 16-page limit
- Compact Qwen diagnostic-ladder figure added; compiled page count remains 16
- Table 3 caption now points readers to the Section 8.1 multilingual KV-prefix failure analysis
- Language clarity pass completed on dense TinyLlama/Qwen diagnosis text; claims and numbers unchanged

## Anonymous Review

- PDF author block: `Anonymous Authors`
- Institute line: `Paper under double-anonymous review`
- No author name, institution, local path, server name, or OpenReview/TokShop mention in the paper text
- TokShop non-archival submission is not cited or used as prior work

## Core Numeric Claims

- Qwen2.5-0.5B tokenizer clean loss: 0.3089
- Phase 2 best clean retention: 2.19x
- Phase 3 distill clean retention: 2.22x
- Phase 4 fixed reconstruction clean retention: 1.84x
- Phase 5 soft prefix p64 clean retention: 1.76x
- Phase 6 KV prefix p32 all-layer clean retention: 1.61x
- Best wins vs tokenizer on Qwen2.5-0.5B: typo light / medium / heavy only
- Qwen2.5-1.5B tokenizer clean loss: 0.3274
- Qwen2.5-1.5B best selected clean retention: 2.52x
- Qwen2.5-1.5B best selected wins: typo medium / heavy only
- TinyLlama tokenizer clean loss: 0.6026
- TinyLlama target-only clean loss: 5.4014
- TinyLlama constant KV clean loss: 0.8907; no input bytes are read
- TinyLlama soft prefix p64 clean retention: 1.26x; wins typo light / medium / heavy
- TinyLlama KV p32 all-layer 3-seed mean clean retention: 0.63x
- TinyLlama KV p32 all-layer seed retentions: 0.69x / 0.66x / 0.55x
- TinyLlama KV p32 all-layer 3-seed mean wins: clean, typo light / medium / heavy, Unicode
- TinyLlama KV p32 all-layer still loses on tokenizer-stress and multilingual buckets
- TinyLlama bucket table added: clean 0.6026 -> 0.3811, Unicode 0.7818 -> 0.6620, tokenizer-stress 0.6057 -> 1.2281, zh 0.5049 -> 10.7154, ar 0.4396 -> 11.1403
- Tokenizer fertility analysis added: TinyLlama uses more input tokens per byte than Qwen2.5-0.5B on clean (0.241 vs 0.218), Unicode (0.493 vs 0.383), code (0.533 vs 0.453), tokenizer-stress (0.700 vs 0.584), Chinese (0.550 vs 0.318), and Arabic (0.547 vs 0.286)
- KV training-curve analysis added: Qwen2.5-0.5B all-layer KV train loss drops 8.61 -> 0.25; TinyLlama seeds drop 10.36/11.28/10.76 -> 0.21/0.30/0.22; TinyLlama constant KV remains much worse than byte-conditioned KV
- All reported LM runs: LLM trainable params 0, checksum delta 0.0

## Framing

- Does not claim ByteBridge succeeds.
- Does not claim fully tokenizer-free LM, because target-side teacher forcing uses Qwen tokenizer.
- States the result as a controlled mixed diagnostic result.
- Keeps Qwen negative claims scoped to Qwen2.5 family.
- States TinyLlama result as selected cross-family evidence, not universal success.
- Includes explicit discussion that TinyLlama-Qwen differences are hypotheses, not established causal explanations.
- Includes target-only and constant-KV controls to address easy-target and generic-soft-prompt explanations.
- Includes tokenizer fertility and optimization-curve diagnostics as partial evidence, not causal proof.
- Related Work now distinguishes ByteBridge from byte-trained models and standard prompt/prefix tuning.
- Limitations now explicitly state that the synthetic/curated benchmark is diagnostic, not a real-world frequency or deployment claim.
- Marks oracle boundary and oracle clean typo as diagnostic upper bounds, not deployable baselines.

## Open Items Before Real Submission

- Add real author names only in EasyChair metadata, not in the anonymous PDF.
- Recheck PRICAI EasyChair fields when submission opens.
- Consider asking chairs only if concerned about simultaneous non-archival TokShop review; PRICAI policy excludes workshops without archival proceedings from the conference/journal dual-submission restriction.
