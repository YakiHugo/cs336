# A1 learning consolidation roadmap

Status snapshot: **implementation and required experiments are complete; scored writeup and
owner-review consolidation remain in progress.** `progress/a1.yaml → current` remains the sole
next-action authority; this file is the human-readable map.

## Position in CS336

```text
CS336 foundations
  A1 Basics       ← current: how text becomes model updates and measurable behavior
  A2 Systems      ← later: make the same work efficient across hardware
  A3 Scaling      ← later: predict quality from model/data/compute budgets
  A4 Data         ← later: construct and improve the training distribution
  A5 Alignment    ← later: deliberately shape model behavior after pretraining
```

A1 supplies the vocabulary used by everything later: tokens, Transformer states, logits, loss,
gradients, optimizer state, training evidence, and architecture/optimization improvements.

## Current project state

| area | implementation / evidence | learning / review |
| --- | --- | --- |
| Tokenizer and data serialization | done; official tests green; TinyStories and OWT tokenizers/data produced | BPE know-how note remains owner review pending |
| Transformer LM | done; official model tests green | core walkthrough completed; full architecture note review pending |
| Loss, AdamW, schedule, clipping | done; official tests green | divergence, warmup, accounting transfers passed; full note review pending |
| Training/checkpoint/generation | done; resume/generation and full runs verified | checkpoint/training-loop note review pending |
| Required experiments | done; curves, generations, ablations, OWT main run, and timed weight tying recorded | most experiment interpretations owner-confirmed; NoPE symmetry correction and generation note remain review debt |
| Scored writeup | §2, §3.5, §4.2, §4.3, and §7 drafts present | final coverage audit and owner review still required |
| Verification | full official suite: 46 passed, 2 platform skips; Ruff green | implementation evidence only, not a proxy for learning |

Large datasets, serialized corpora, run logs, and checkpoints remain local and gitignored. The git
checkpoint contains source code, configurations, learning notes, plots, progress, and journal state.

## Segment-first review route

### Segment 1 — End-to-end model and training architecture

Map and teach the complete path before exercises:

```text
text → BPE token IDs → batches/targets → embeddings → Transformer blocks
     → logits → cross-entropy → gradients → clipping → AdamW/schedule
     → checkpoint/evaluation/generation → evidence about behavior and quality
```

The explanation must connect shapes, causal information flow, residual/norm behavior, optimization
state, and the distinction between train loss, validation loss, and generated behavior. After the
owner explicitly pushes forward, create one bounded worksheet under `notes/a1/reviews/` and review
the completed file as a whole.

### Segment 2 — Improvement levers and experimental evidence

Connect RMSNorm, pre/post-norm, RoPE/NoPE, SwiGLU/SiLU, decoding, precision, batch/LR choice, and
weight tying to stability, quality, cost, and what each controlled experiment supports or rules out.
Then use one concentrated worksheet and whole-file review.

### Segment 3 — Resource and artifact closure

Consolidate parameter/FLOP/memory/MFU reasoning, resolve remaining know-how corrections, audit every
scored handout deliverable, and finish the A1 writeup. A1 learning is complete only after the owner
reviews the resulting notes/worksheet; green tests and agent-written drafts alone are insufficient.
