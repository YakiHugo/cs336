# Loss and optimization — know-how & decisions

Learning status: **review_pending** — agent draft. The owner correctly identified confident
wrong predictions as higher-loss and excessive learning rate as the first suspect for immediate
divergence. Hyperparameter/failure transfer should be revisited when reading real loss curves.
The §4.3 AdamW memory-accounting subsection is **owner-confirmed** after the owner correctly
converted parameter counts to bytes and transferred the fixed-versus-activation distinction to
gradient accumulation.

Code: `assignments/assignment1-basics/cs336_basics/optim.py`,
`assignments/assignment1-basics/cs336_basics/training.py`  
Adapters: §4 adapters remain thin  
Tests: `test_nn_utils.py`, `test_optimizer.py` — **5 passed**

Sources: `docs/handouts/cs336_assignment1_basics.md:1280-1335` (PDF pp. 28–29,
§4.1); `docs/handouts/cs336_assignment1_basics.md:1336-1562` (PDF pp. 29–34,
§4.2–4.5).

## Pipeline (mental model)

```text
batch inputs (B,T) → model logits (B,T,V)
shifted targets (B,T) → stable cross-entropy → scalar loss
loss.backward() → one gradient tensor per used parameter
global gradient clipping → AdamW moment/decay update at scheduled LR
zero gradients → next step
```

## What each mechanism controls

| Mechanism | Role | Failure signal |
| --- | --- | --- |
| Cross-entropy | `-log p(correct next token)` averaged over positions | confident wrong predictions are penalized most |
| Learning rate | global optimizer step scale, not merely "speed" | too high: spikes/divergence/NaN; too low: slow budget use |
| Adam first moment `m` | smooth recent gradient direction | too much memory can respond slowly to change |
| Adam second moment `v` | estimate gradient scale for adaptive steps | stale/noisy scale estimates destabilize effective steps |
| Weight decay | independently shrink parameters each step | too high suppresses capacity; too low weakens regularization |
| Warmup | ramp safely while model and moments are immature | absent/short warmup can destabilize early steps |
| Cosine decay | transition from large learning steps to refinement | wrong resume step silently changes the intended schedule |
| Global clipping | rescale an occasional oversized gradient without changing its direction | repeated clipping indicates LR/data/numerics need diagnosis |

## AdamW memory: fixed state versus batch-dependent activations

Source: `docs/handouts/cs336_assignment1_basics.md:1475-1532` (PDF pp. 32–33,
§4.3 `adamw_accounting`). A float32 scalar occupies 4 bytes. If the model has `P` parameters,
training persistently holds one parameter array, one gradient array, and AdamW's two moment arrays:

```text
parameters:       4P bytes
gradients:        4P bytes
AdamW m and v:    8P bytes
fixed subtotal:  16P bytes
```

This is four times the memory needed merely to load float32 weights. Activations are separate: they
store intermediate forward results needed by backward and grow with the physical microbatch. Gradient
accumulation increases the effective batch by processing microbatches sequentially, so it does not
multiply peak activation memory by the accumulation count.

For the handout's simplified inventory, with batch `B`, sequence `T`, layers `L`, width `D`, heads
`H`, vocabulary `V`, and `F=d_ff`, the stored activation scalar count is

`A = BL(8TD + 4TF + 2HT²) + B(TD + 2TV)`.

The first term covers two block RMSNorms, Q/K/V, both attention maps, the value/output projections,
and the five SwiGLU stages; the final term covers final RMSNorm, logits, and cross-entropy's logit-sized
intermediate. With `F=8D/3`, total handout-accounted peak memory is `16P + 4A` bytes. Real frameworks
may fuse, recompute, or retain different buffers, so this is an explicit accounting model rather than
an allocator guarantee.

## Decisions shipped

1. Cross-entropy uses the log-sum-exp cancellation directly from logits; no explicit
   probability or `log(0)` path.
2. AdamW follows handout Algorithm 1: decoupled parameter shrink, first/second moment
   updates, and bias-corrected step size.
3. Clipping computes a single norm across every existing parameter gradient and scales
   them together; frozen or unused parameters with `grad=None` are skipped.
4. The scheduler implements linear warmup, cosine decay through `T_c`, then the minimum LR.

## Invariants and failure checklist

1. Targets are token IDs with the same leading axes as logits and no vocabulary axis.
2. Cross-entropy consumes logits directly and averages all batch-like positions.
3. Gradients accumulate by default; failure to clear them changes the effective update.
4. Clip after backward and before AdamW so moment estimates see the clipped gradient.
5. Checkpoint resume must restore AdamW `m/v/step` and scheduler iteration, not only weights.
6. An isolated spike is a clipping use case; sustained divergence calls for LR, data,
   precision, and implementation diagnosis rather than a larger clipping threshold.
7. Convert tensor counts to float32 memory by multiplying scalar count by 4 bytes; do not confuse
   parameter count with bytes or GB.
8. AdamW's fixed fp32 training state is four parameter-sized arrays (`θ,g,m,v`); batch size changes
   activations, not this fixed subtotal.
9. Gradient accumulation reuses fixed state and one microbatch's activations; it increases effective
   batch and work per optimizer update, while the optimizer/scheduler advance only after accumulation.

## Review debt

- Owner correction/confirmation of this draft.
- Revisit the failure map using observed train/validation curves from TinyStories runs.
