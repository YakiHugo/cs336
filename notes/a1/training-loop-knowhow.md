# Training loop and resume — know-how & decisions

Learning status: **review_pending** — agent draft. The owner correctly diagnosed diverging
train/validation curves as overfitting and recognized that failure to memorize one fixed
minibatch points to the training chain rather than normal generalization difficulty.

Code: `assignments/assignment1-basics/cs336_basics/train.py`,
`assignments/assignment1-basics/cs336_basics/training.py`  
CLI: `uv run cs336-train`  
Verification: official suite **46 passed, 2 platform skips**; single-batch and resume smoke
checks pass. Full TinyStories run: **5,000 steps / 40.96M tokens / validation loss 1.6903**.

Sources: `docs/handouts/cs336_assignment1_basics.md:1563-1677` (PDF pp. 34–37,
§5); `docs/handouts/cs336_assignment1_basics.md:1750-1848` (PDF pp. 38–40,
§7.1–7.2); `docs/handouts/cs336_assignment1_basics.md:1866-1875` (PDF p. 41,
batch-size experiment).

## Pipeline (mental model)

```text
memmapped token IDs
  → random x and one-token-shifted y
  → zero gradients
  → logits and cross-entropy
  → backward
  → observe global gradient norm, then clip
  → set scheduled LR and AdamW step
  → log train evidence

periodically:
  deterministic validation batches under no_grad
  → average loss
  → save best / periodic / last checkpoints
```

## Loss-curve failure map

| Evidence | First interpretation | Next discriminating check |
| --- | --- | --- |
| train and validation decrease | healthy learning/generalization | continue; compare at equal tokens and wall time |
| train decreases, validation turns upward | overfitting | select best-validation checkpoint; inspect data/regularization/budget |
| both stay high/flat | underfitting, weak LR, or broken chain | fixed-minibatch overfit test |
| both spike or become NaN | optimization/numerical instability | LR, gradient/activation norms, precision, implementation |
| isolated gradient spike | hard batch/outlier | clipping should contain it; inspect recurrence |
| resume causes immediate discontinuity | missing/mismatched state | optimizer moments, step/schedule, RNG, config |

## Decisions shipped

1. Train and validation batch RNGs are separate. Validation reuses the same sampled batches
   at every evaluation so curve changes reflect the model rather than sample churn.
2. JSONL records step, train/validation loss, LR, pre-clip gradient norm, step time,
   tokens/s, and cumulative wall time.
3. Checkpoints include weights, AdamW state, completed iteration, configs, training-sampler
   RNG, torch RNG, best validation loss, and elapsed wall time.
4. Resume rejects changes to batch size, LR schedule, AdamW, clipping, or seed. Extending
   max steps and changing log/eval/checkpoint cadence remain allowed.
5. `.npy` token arrays load with `mmap_mode='r'`; raw binary arrays use `np.memmap` with
   explicit dtype. Large corpora do not need to fit in RAM.
6. `best.pt` represents the best observed validation model; `last.pt` represents resumable
   run state. They answer different questions and must not be conflated.

## Smoke evidence (2026-08-02)

- Tiny 1-layer model, one fixed minibatch: cross-entropy **4.102376 → 0.000080**
  in 150 updates. This rules out a systemic break in target alignment, forward/backward,
  clipping, or AdamW update.
- Continuous 8 steps versus 4 steps + save/load + 4 steps: maximum parameter difference
  **0.0**. Optimizer-state resume is trajectory-equivalent for the deterministic smoke.
- Full CLI on memory-mapped synthetic train/validation arrays produced decreasing
  validation loss plus `metrics.jsonl`, `best.pt`, periodic checkpoints, and `last.pt`.

## Real TinyStories sweep evidence (2026-08-02)

The controlled 100-step sweep used the handout base architecture, batch size 32, context
length 256, seed 42, 10 warmup steps, cosine decay ending at step 100, and the same four
fixed validation batches. Only the maximum/minimum learning rate changed.

| max LR | min LR | validation loss at step 100 | interpretation |
| ---: | ---: | ---: | --- |
| `1e-5` | `1e-6` | 8.5960 | stable but far too slow |
| `1e-4` | `1e-5` | 5.7391 | stable, still conservative |
| `3e-4` | `3e-5` | 4.2651 | healthy |
| `1e-3` | `1e-4` | 3.6256 | healthy, near the short-run optimum |
| `3e-3` | `3e-4` | **3.5892** | narrow short-run winner |
| `1e-2` | `1e-3` | 4.1426 | crossed the optimum; gradient spike 7.07 |
| `3e-2` | `3e-3` | 4.6333 | early overshoot; clipping contained but did not cure it |
| `1e-1` | `1e-2` | 5.5335 | severe overshoot and practical non-convergence |
| `3e-1` | `3e-2` | 5.8288 | validation diverged to 24.20 near peak LR; decay enabled partial recovery |

The `3e-2` run's step-10 training loss reached 10.8013 and its pre-clip gradient norm
reached 5.8446. Its later recovery demonstrates that “did not become NaN” is weaker evidence
than “used the token budget efficiently.” The completed logarithmic bracket places the
100-step optimum around `1e-3`–`3e-3`, but `3e-3` beats `1e-3` by only 0.0364 loss. A
schedule compressed into 100 steps does not prove that the narrow winner has the best or
safest 5,000-step endpoint. The already-running `1e-3` trajectory reached validation loss
2.0179 at step 1600 without instability, so preserving that full-run evidence is preferable
to restarting solely on the tiny short-run difference.

“Divergence” does not require NaN. The `3e-1` curve provides the clear boundary evidence:
at step 10 training loss was 54.12, at step 20 the pre-clip gradient norm was 22.27, and at
step 25 validation loss was 24.20 versus its 9.25 initialization. It recovered only when the
schedule moved LR far below its peak. Gradient clipping bounded each update but could not
turn a destructive step size into efficient optimization. The “edge” is therefore a regime
shaped by schedule, clipping, and model state, not a universal scalar threshold.

## Full-run evidence (2026-08-02)

The Apple Silicon configuration used batch 32, context 256, 5,000 updates, max LR `1e-3`,
200 warmup steps, and cosine decay to `1e-4` at exactly step 5,000. It therefore processed
40,960,000 tokens, matching the handout's low-resource prescription.

- Validation loss: **9.2500 → 1.6903**.
- The relaxed target of ≤2.00 was first crossed at step 1,800 (1.9818).
- Every later fixed-validation measurement improved; the final checkpoint is also the best.
- Trajectory wall time: **3,264 s (54.4 min)**, excluding the paused sweep experiments and
  the discarded post-checkpoint work before resume.
- The cosine LR fell from `1e-3` to `1e-4`; pre-clip gradient norms remained roughly
  0.4–0.5. Random single-batch training losses oscillated while the 20-batch validation mean
  declined smoothly, showing why isolated training-loss points are weak model-selection
  evidence.

## Batch-size experiment (2026-08-03)

The controlled comparison fixes total tokens, not updates:

```text
tokens = batch_size × steps × context_length = 2,097,152
```

Validation independently uses four fixed batch-32 samples for every run. Initial peak LR
was scaled linearly from `1e-3` at B32, then B64/B128/B256 were re-run at lower LR where
necessary.

| batch | steps | selected peak LR | final validation | wall s | effective tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8192 | `3.125e-5` | 3.551 | 333.8 | 6.28k |
| 8 | 1024 | `2.5e-4` | 3.005 | 210.8 | 9.95k |
| 32 | 256 | `1e-3` | **2.907** | 173.2 | 12.11k |
| 64 | 128 | `2e-3` | 3.212 | **165.4** | **12.68k** |
| 128 | 64 | `2e-3` | 3.658 | 178.6 | 11.74k |
| 256 | 32 | `4e-3` | 4.349 | 206.3 | 10.16k |

Mental model:

1. Averaging more examples reduces gradient variance, so B1 showed gradient norms around
   2–3 and large single-batch loss swings while B32 was around 0.4.
2. At equal tokens, larger batch means proportionally fewer optimizer updates. LR scaling
   can increase distance per update only until the stability boundary intervenes.
3. Hardware throughput improved through B32–B64, then fell: B256 used 38.5 GiB of MPS
   driver memory, above its 37.4 GiB recommended maximum, and became bandwidth/memory bound.
4. B32 is the best quality point; B64 is only 4.7% faster but costs 0.304 validation loss.
   B128 and B256 are dominated on both quality and elapsed time.
5. Lowering LR repaired some B128/B256 overshoot but did not erase the deficit, ruling out
   LR as the sole cause. The useful or critical batch region for this model/budget/device is
   around 32–64, not a universal constant.

## Invariants and failure checklist

1. `y` is exactly `x` shifted one token; document delimiter tokens remain ordinary targets.
2. Zero gradients before each ordinary update unless gradient accumulation is intentional.
3. Clip after backward and before AdamW; log the pre-clip norm so clipping does not hide spikes.
4. Validation performs no gradient/update and compares consistent data samples.
5. A fixed minibatch should be memorized rapidly; failure localizes the problem below
   generalization and data diversity.
6. Resume at completed iteration `k` means the next LR is schedule step `k`, with AdamW
   moments and batch RNG restored.
7. Choose a model by validation evidence, not the final or lowest training loss alone.

## OpenWebText scale-up notes (2026-08-16, active run)

The OWT tokenizer has 32K outputs rather than TinyStories' 10K. The final logits tensor therefore
costs about 3.2 times as much memory at the same `(B,T)`, even though the Transformer hidden states
are unchanged. Two implementation transformations preserve the intended optimization:

1. Gradient accumulation with physical batch 16 and two microbatches computes the mean gradient of
   the effective batch 32, then clips and calls AdamW exactly once. It preserves 8,192 tokens per
   optimizer step and avoids treating two microbatches as two schedule steps.
2. Chunked LM-head loss computes the same token-summed cross-entropy in pieces, accumulates the
   gradient with respect to the final hidden states and output projection, then backpropagates through
   the Transformer once. This frees each large `(tokens, 32000)` logits chunk before constructing the
   next one; it changes peak memory, not the loss definition.

Numerical checks against the ordinary full-logits/full-batch step found loss differences below
`1.5e-7` and maximum parameter differences below `1.7e-6`, consistent with floating-point reduction
order. These are engineering equivalences, not architectural modifications.

OWT/TinyStories loss interpretation is a separate invariant: per-token cross-entropy depends on both
the data distribution and tokenizer. OWT is more diverse and has a 32K vocabulary, so its raw loss is
expected to be higher. Initial random losses provide a sanity check (`ln(32000) ~= 10.37` versus
`ln(10000) ~= 9.21`), but final raw losses from the two tokenizations do not form a direct quality
ratio. Within-run validation trend and sample quality answer whether the OWT run is learning.

### Reusable MPS execution profile

The laptop's low-power mode is intentional, so the reusable profile optimizes software execution
without treating power mode as a variable. `training_profiles.toml` centralizes model shape,
precision, effective batch construction, optimizer/schedule, evaluation, checkpoint cadence, paths,
and seed. CLI arguments override only the variable under study. `scripts/run_a1_training.sh` adds an
idle-sleep assertion while leaving the chosen power policy untouched.

BF16 autocast keeps parameters and AdamW state in FP32 while running eligible matrix-heavy forward
operations in BF16. It is not assumed equivalent merely because it is faster: the exact step-200 OWT
checkpoint was continued with the same RNG, batches, LR, and optimizer moments. At step 250, BF16 loss
was `6.29182` versus FP32 `6.29156`, gradient behavior matched, validation reached `6.12567`, and
steady throughput was about 10.3K tokens/s. FP16 was rejected because its gradients rapidly collapsed
in the same representative test. MPS fast-math/Metal environment flags were also rejected as defaults
because their apparent gain did not reproduce after reversing benchmark order.

The full OWT run completed all 5,000 updates with validation loss `10.3862 -> 4.6485`; every
200-step validation point improved, so the final checkpoint is also the best. Generated evidence
adds information the curve cannot: sampling produced recognizable web/news document scaffolding but
fabricated entities and incoherent claims, while greedy decoding entered repetition loops. Falling
validation loss therefore proves improved next-token prediction on OWT, not factuality or long-range
semantic competence.

Real-device resume caveat: loading a whole checkpoint with `map_location="mps"` also maps
saved RNG-state byte tensors to MPS, while CPU generators require CPU state. Move RNG states
back to CPU before `set_state`; model and optimizer state still belong on the training device.
After this fix, the resumed full run reproduced its earlier step-1525 loss exactly
(2.0825698376).

## Review debt

- Owner correction/confirmation of this draft.
- Revisit generation quality as the remaining §7.2 evidence arrives.
