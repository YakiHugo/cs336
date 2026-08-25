# Transformer LM architecture — know-how & decisions

Learning status: **review_pending** — agent draft. The owner completed a from-first-principles
pass and correctly reasoned about lookup semantics, causal context, attention-vs-FFN roles,
next-token targets, confident errors, and `T²` cost. RoPE translation invariance and the exact
RMSNorm scale cancellation required correction and should be rechecked during review. The
RMSNorm ablation and §7.5 weight-tying subsections are now **owner-confirmed** after successful
transfer checks. §3.5 resource accounting is also **owner-confirmed**: the owner distinguished
context-quadratic attention mixing from width-quadratic projections/FFNs in a transfer case.

Code: `assignments/assignment1-basics/cs336_basics/nn.py`  
Adapters: all §3 model adapters remain thin  
Tests: `test_model.py` — **13 passed**; full official suite — **46 passed, 2 platform skips**

Sources: `docs/handouts/cs336_assignment1_basics.md:552-638` (PDF pp. 13–14,
§3.1–3.2); `docs/handouts/cs336_assignment1_basics.md:774-914` (PDF pp. 17–20,
§3.3–3.4.1); `docs/handouts/cs336_assignment1_basics.md:919-1226` (PDF
pp. 21–27, §3.4.2–3.5); `docs/handouts/cs336_assignment1_basics.md:1876-1908`
(PDF pp. 41–42, §7.2 generation).

## Pipeline (mental model)

For `B=batch`, `T=sequence`, `D=d_model`, `V=vocab`, and `H=heads`:

```text
token IDs                    (B, T)
  → token embedding lookup    (B, T, D)
  → Transformer block × L    (B, T, D)  [shape-preserving]
      attention scores       (B, H, T, T)  [important internal exception]
  → final RMSNorm             (B, T, D)
  → LM-head linear            (B, T, V)  logits
  → softmax                   (B, T, V)  probabilities
```

Training uses the prediction at every `T` position against the next-token targets.
Autoregressive generation normally consumes only the final position, `(B, V)`, to choose
one next token.

## Resource accounting: parameters, token-wise compute, and pairwise compute

Source: `docs/handouts/cs336_assignment1_basics.md:1228-1291` (PDF pp. 27–28,
§3.5 `transformer_accounting`). For one sequence, the matrix-multiply FLOPs are:

| operation | matrix shape | FLOPs |
| --- | --- | ---: |
| Q, K, V projections per layer | three `(T,D)@(D,D)` | `6TD²` |
| attention scores per layer | head-batched `(T,d_h)@(d_h,T)` | `2T²D` |
| attention-weighted values per layer | head-batched `(T,T)@(T,d_h)` | `2T²D` |
| attention output projection per layer | `(T,D)@(D,D)` | `2TD²` |
| SwiGLU per layer | two `(T,D)@(D,F)` plus `(T,F)@(F,D)` | `6TDF` |
| vocabulary projection | `(T,D)@(D,V)` | `2TDV` |

Thus the counted forward-pass total is
`L(8TD² + 4T²D + 6TDF) + 2TDV`. Embedding lookup is indexing, while RMSNorm, RoPE,
softmax, and element-wise gates are not matrix multiplications and are excluded by this particular
accounting convention—not literally free in a real profiler.

Only the two attention-mixing products contain `T²`. Projections, the FFN, and vocabulary logits
are linear in `T`. Therefore multiplying context length by `c` makes the complete forward cost fall
between `c×` and `c²×`; it approaches quadratic growth only when pairwise attention dominates.
Conversely, at fixed context and vocabulary, increasing `D` makes the `D²` block projections/FFN
grow faster than the `T²D` attention-mixing and `DV` vocabulary terms.

## Autoregressive decoding

Generation repeatedly performs the same closed loop:

```text
current token prefix
  → model logits at the final position `(V,)`
  → decoding policy chooses one token ID
  → append that ID to the prefix
  → stop at EOT / token budget; otherwise repeat
```

- **Greedy** always takes `argmax(logits)`. With fixed weights and prompt it is deterministic,
  but one locally likely choice can lock the whole continuation onto a bland or mistaken path.
- **Temperature** samples from `softmax(logits / temperature)`. Values below 1 sharpen the
  distribution; values above 1 flatten it. It changes selection, not model knowledge.
- **Top-p** sorts token probabilities and keeps the smallest prefix whose cumulative mass
  reaches `p`, then renormalizes and samples. The candidate count adapts to model confidence.
- A fixed random seed makes the same sampling operations reproducible on the same setup; a
  different seed explores a different path without changing model quality.
- Once the prefix exceeds the trained context length, this implementation passes only its
  latest `T` tokens. Earlier tokens can no longer directly influence the next-token logits.

Empirically, the 40.96M-token checkpoint produced grammatical short stories and learned EOT.
Conservative sampling (`temperature=0.5`, `top_p=0.9`) was concise and coherent but formulaic.
More open sampling (`0.8`, `0.9`) admitted a fluent but repetitive “powder” story with semantic
errors. Tightening top-p to `0.5` recovered a mostly coherent magic-ball story, though repeated
phrases remained. Greedy produced a complete story but reversed who had lost the toy at the end.
These are model-distribution errors exposed by decoding, not proof that decoding itself failed.

## RMSNorm ablation: scale control versus smaller updates

Source: `docs/handouts/cs336_assignment1_basics.md:1909-1925` (PDF p. 42,
§7.3 `layer_norm_ablation`). The pre-norm block is:

```text
z = x + Attention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

The residual path still carries `x` unchanged, while RMSNorm presents each learned sublayer
with an approximately unit-RMS input. Consequently, an upstream change in residual magnitude
does not proportionally rescale every downstream Q/K/V and FFN input. Removing RMSNorm does
not change tensor shapes or causal information flow; it removes this scale boundary.

The failure chain is:

```text
residual magnitude drifts
  → Q/K scores and FFN activations become scale-sensitive
  → logits/loss and gradients can become very large
  → a formerly useful LR makes destructive updates
```

This is optimization instability, not necessarily literal floating-point overflow. Gradient
clipping can keep parameters finite, but if the raw norm is thousands or millions and every
step is clipped to 1, update direction and effective optimization are still badly distorted.
A lower LR reduces update distance and can restore a trainable curve; it does not restore the
invariant that every sublayer receives a controlled-scale input.

Controlled evidence (same seed/data, B32, T256, 100-step cosine schedule):

| architecture / peak LR | validation loss at 0 | 25 | 50 | 75 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-norm / `3e-3` | 9.249 | 4.668 | 4.012 | 3.722 | 3.589 |
| no norm / `3e-3` | 16.170 | 2,322 | 6,559 | 197,508 | 63,151 |
| no norm / `1e-3` | 16.170 | 4.926 | 4.050 | 3.708 | **3.556** |
| no norm / `3e-4` | 16.170 | 6.082 | 4.799 | 4.407 | 4.279 |

At identical initialization, block-output RMS grew `1.08→1.18→1.28→1.40` with pre-norm,
but `1.08→1.26→1.93→14.53` without normalization; final logit RMS was `0.31` versus `4.48`.
At no-norm `3e-3`, the raw gradient norm reached 705 by step 20 and 19.7 million by step 50.
Lowering to `1e-3` stabilized the run; lowering further to `3e-4` merely slowed learning in
this budget. RMSNorm therefore enlarged the usable LR range rather than being required for
every finite run of this shallow model.

## Pre-norm versus post-norm

Source: `docs/handouts/cs336_assignment1_basics.md:1922-1942` (PDF pp. 42–43,
§7.3 `pre_norm_ablation`). For a generic residual sublayer `F`:

```text
pre-norm:   y = x + F(Norm(x))
post-norm:  y = Norm(x + F(x))
```

Their backward paths differ even though both control forward scale:

```text
pre:   dy/dx = I + J_F J_Norm
post:  dy/dx = J_Norm (I + J_F)
```

Pre-norm always retains an explicit identity term through the residual stream. In post-norm,
the residual branch itself passes through each normalization Jacobian; stacking many blocks
repeatedly transforms the backward signal. This makes post-norm more sensitive to depth,
learning rate, initialization, and warmup. It does **not** imply every post-norm model must
have vanishing gradients or fail: an initial 4-layer diagnostic here showed finite, comparable
block-parameter gradients in both variants.

Controlled 100-step evidence:

| architecture / peak LR | validation loss at 0 | 25 | 50 | 75 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-norm / `3e-3` | 9.249 | 4.668 | 4.012 | 3.722 | **3.589** |
| post-norm / `3e-3` | 9.247 | 5.964 | 5.881 | 5.854 | 5.837 |
| pre-norm / `1e-3` | 9.249 | 4.546 | 3.942 | 3.713 | 3.626 |
| post-norm / `1e-3` | 9.247 | 4.749 | 3.982 | 3.712 | **3.593** |

Post-norm did not diverge at `3e-3`; it entered a poor plateau. Lowering the peak LR to
`1e-3` recovered a curve essentially equal to the best pre-norm endpoint. The supported claim
for this shallow/budgeted experiment is therefore **narrower LR robustness**, not inevitable
post-norm failure or observed gradient vanishing.

Warmup is a schedule segment, not another normalization mechanism. In this assignment
(`docs/handouts/cs336_assignment1_basics.md:1518-1537`, PDF p. 33), for `t < T_w`:

```text
LR(t) = (t / T_w) × LR_max
```

It protects the early random-initialization/immature-optimizer-state phase by approaching the
peak gradually. A longer warmup delays exposure to a high peak; it does not make an inherently
unsafe `LR_max` safe once the schedule reaches it. Lowering the peak changes the update scale
throughout the high-LR region.

## RoPE versus NoPE

Source: `docs/handouts/cs336_assignment1_basics.md:1943-1953` (PDF p. 43,
§7.3 `no_pos_emb`). NoPE removes the Q/K rotations but retains the lower-triangular causal
mask. It therefore removes explicit relative-position coordinates, not all structural clues
about order.

The distinction appears across depth:

1. In the first NoPE attention layer, the last token reads earlier token embeddings as an
   unordered content collection. For `[A,B,C]` and `[B,A,C]`, the final query sees the same
   keys/values in a different enumeration, so attention's weighted sum is permutation-invariant.
2. Earlier positions do not have the same histories: A/B representations are formed from
   different allowed prefixes. A second layer can read these history-bearing representations,
   indirectly recovering some order.
3. RoPE makes Q/K compatibility depend on relative angle immediately, so even the first layer
   can distinguish the two arrangements and express distance-sensitive routing directly.

A random-model probe confirmed this mechanism at the final position:

| position scheme | layers | max logit difference, `[A,B,C]` vs `[B,A,C]` |
| --- | ---: | ---: |
| NoPE | 1 | `0.00000030` (floating-point noise) |
| NoPE | 2 | `0.6378` |
| RoPE | 1 | `0.0906` |

Controlled training at peak LR `3e-3`:

| position scheme | validation 0 | 25 | 50 | 75 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RoPE | 9.249 | 4.668 | 4.012 | 3.722 | **3.589** |
| NoPE | 9.250 | 4.566 | 4.084 | 3.907 | 3.825 |

NoPE learned most TinyStories regularities and was briefly ahead at step 25, ruling out the
claim that it is simply order-blind or untrainable. RoPE pulled ahead as learning progressed,
ending 0.236 loss lower. Explicit position lets capacity be spent on language rather than
reconstructing order from nested prefixes. The likely weaknesses of NoPE are exact distance,
counting, copying, and repeated-token order, not basic local fluency. RoPE adds no learned
parameters and does not change the dominant `T²` attention cost.

Boundary case: for a prefix consisting only of identical token representations and no distinct
boundary token, NoPE cannot infer whether the current token is the 3rd or 30th occurrence.
Every Q/K/V is identical; softmax may assign `1/3` versus `1/30`, but the weighted average of
identical V vectors is identical. Residual and position-wise FFN operations preserve equality,
so the symmetry holds inductively through all layers. A distinct BOS/content token can break
the symmetry and leak count through changing mixture proportions; floating-point noise is not
a reliable positional signal. RoPE breaks the symmetry directly by rotating Q/K differently.

## SwiGLU versus SiLU at matched parameter count

Source: `docs/handouts/cs336_assignment1_basics.md:1954-1974` (PDF pp. 43–44,
§7.3 `swiglu_ablation`). The compared FFNs are:

```text
SwiGLU(x) = W2(SiLU(W1x) ⊙ W3x)
SiLUFFN(x) = W2(SiLU(W1x))
```

SwiGLU computes two independent views of the same contextual token vector. The activated
`W1x` branch acts as an input-dependent gate on the content branch `W3x`; elementwise
multiplication can suppress, pass, or rescale each intermediate feature before projection back
to `D`. An ordinary SiLU FFN nonlinearly transforms one view but lacks this learned
multiplicative interaction.

Equal `d_ff` would be confounded because SwiGLU has three matrices and SiLU only two. Ignoring
biases, the handout's widths approximately match both FFN parameter counts at `8D²`:

```text
SwiGLU: 3 × D × (8/3 D) ≈ 8D²   (implemented d_ff=1344 for D=512)
SiLU:   2 × D × (4D)     = 8D²   (implemented d_ff=2048)
```

Measured counts and controlled curve:

| FFN | one-block FFN params | total model params | validation 0 | 25 | 50 | 75 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SwiGLU | 2,064,384 | 22,696,448 | 9.249 | 4.668 | 4.012 | 3.722 | **3.589** |
| SiLU | 2,097,152 | 22,827,520 | 9.269 | 4.999 | 4.248 | 3.932 | 3.782 |

SiLU had 0.58% more total parameters yet finished 0.192 loss worse. This supports better
parameter efficiency from multiplicative gating for this TinyStories model, seed, and short
budget. It does not establish universal superiority across seeds, training horizons, data, or
scale; those broader claims require repeated experiments.

## Why each implemented component exists

| Component | Transformation | Purpose | Shape/cost fact |
| --- | --- | --- | --- |
| Embedding | integer ID → learned row vector | Gives each vocabulary item a trainable continuous representation | `(B,T) → (B,T,D)`; table is `(V,D)` |
| Linear | `x Wᵀ` | Changes the feature vocabulary at each independent outer position | `(...,d_in) → (...,d_out)` |
| RMSNorm | `x / sqrt(mean(x²)+eps) * g` | Stabilizes activation scale while preserving direction information and the residual-stream shape | Normalizes only final `D`; `(...,D) → (...,D)` |
| Attention | content-addressed weighted read over allowed positions | Moves information between tokens; causal mask prevents future leakage | scores `(B,H,T,T)`; main sequence-dependent quadratic cost |
| RoPE | rotate Q/K coordinate pairs by position-dependent angles | Adds relative-position signal to routing without changing values | same shape; shared shifts preserve relative angle |
| SwiGLU FFN | gated `D → d_ff → D` transformation | Nonlinearly processes the context already present at each position | position-wise; cannot independently observe another token |
| Transformer block | pre-norm attention + FFN inside residual additions | Alternates cross-position communication with local feature processing | Public shape `(B,T,D)` is preserved so residual addition is legal |
| LM head | hidden vector → one score per token | Turns each contextual hidden state into next-token logits | `(B,T,D) → (B,T,V)` |

## Decisions shipped

| Decision | Implementation | Reason/invariant |
| --- | --- | --- |
| Embedding layout | `weight[vocab_size, d_model]`; direct indexing | Token IDs select rows and arbitrary leading input axes are retained |
| RMS reduction axis | `mean(dim=-1, keepdim=True)` | Each token vector is normalized independently; broadcasting preserves shape |
| RMS arithmetic | Input is upcast to float32 before squaring, then result is cast back | Prevents half-precision overflow as required by §3.4.1 |
| Attention layout | project once per Q/K/V, split `D` into `H×d_head`, batch heads | Every head sees all positions; feature channels, not the sequence, are split |
| Causal mask | one lower-triangular `T×T` boolean mask broadcast across batches/heads | Position `i` reads only keys `j≤i`; parallel training cannot leak targets |
| RoPE buffers | precompute float32 sin/cos; apply to Q/K only, not V | Positions affect matching while V remains the content payload |
| Block state | `x += attn(norm(x)); x += ffn(norm(x))` | Clean residual path, stable sublayer scale, shape-safe updates |
| Adapters | Instantiate module, load supplied state dict, call forward | All model logic stays in `cs336_basics/` |

## §7.5 improvement: tied input/output embeddings (owner-confirmed)

Input embedding and output classification attach different operations to the same vocabulary rows:
lookup selects row `E[token]`, while the LM head computes all dot products `h Eᵀ`. Sharing `E` removes
one 32K x 512 table and lets occurrences train a token row from both its input and target roles. It
does **not** remove the `D x V` output projection or any Transformer block computation. On OWT this
reduced unique parameters from 45.22M to 28.84M and substantially reduced dense gradient/Adam-state
memory traffic, while vocabulary-logit arithmetic stayed fixed.

One table must satisfy two scale requirements. The assignment's ordinary embedding std of 1 is safe
for lookup followed by RMSNorm but far too large for a 32K-way classifier. Reinitializing the shared
table at std 0.02 fixes output logits but makes the residual-stream input much smaller. An input-only
fixed multiplier of `sqrt(D)` separates these effective scales without duplicating weights—the same
pattern used in the original Transformer. Empirically, unscaled tied variants ended around 6.91-6.94
validation loss at 100 OWT steps, worse than untied 6.75; std 0.02 plus `sqrt(512)` scaling reached
6.594, and LR 2e-3 improved it to 6.533. This is evidence for the complete scale-aware modification,
not weight sharing in isolation.

The formal 5,000-step OWT run reached final/best validation loss **4.43271** in 2,678.87 seconds
(44.65 minutes); every one of its 26 validation measurements improved. The untied 5,000-step main
run reached 4.64853. At the same LR `1e-3`, the controlled 100-step comparison was 6.594 tied versus
6.750 untied; tuning the tied run to LR `2e-3` then improved it to 6.533. The same-LR short comparison
isolates a per-update/statistical benefit because all settings except the architecture match. The
timed curve instead measures the deployable combination of architecture, tuned hyperparameters, and
reduced parameter/optimizer traffic. These answer different questions and should not be substituted.

## Invariants and failure checklist

1. `B` and `T` are leading batch-like lanes for position-wise operations; they must not
   accidentally be contracted by Linear, Embedding, RMSNorm, or FFN.
2. Transformer block input and output are both `(B,T,D)`. A residual addition failing is
   evidence that some sublayer leaked an internal dimension.
3. Only attention mixes sequence positions. Its score/mask axes are `(...,T_query,T_key)`;
   causal masking must forbid future keys, not whole feature channels.
4. Attention's `T×T` score matrix explains quadratic sequence-time and memory growth. A
   position-wise `(B,T,D)` tensor alone does not.
5. LM-head output is logits, not normalized probabilities. Cross-entropy should consume
   logits directly; softmax is needed for interpreting/sampling probabilities.
6. RMSNorm must reduce over `D`, retain that axis for broadcasting, add epsilon, and square
   in float32. Wrong-axis reduction couples unrelated tokens or batches; low-precision
   squaring can produce `inf`/`NaN`.
7. An embedding lookup appends `D` to the ID tensor's full shape. If `T` disappears, the
   implementation indexed or reshaped the lookup table incorrectly.
8. Generation selects from only the final-position logits, appends exactly one token per
   iteration, and stops before decoding EOT. Greedy must replay identically; stochastic claims
   require the seed and sampling settings to be recorded.
9. Finite loss plus active gradient clipping is not evidence of a stable trajectory. Inspect
   raw pre-clip norms and validation curves; clipping can hide a severely unstable scale regime.
10. Do not infer “post-norm gradients vanished” from a poor curve alone. Compare per-layer
    gradients or test LR sensitivity; identity-path reasoning is a depth/stability mechanism,
    not proof of one specific failure in every finite model.
11. A causal mask supplies asymmetric prefix structure, but not a unique position label. With
    NoPE, do not claim complete order blindness; equally, do not assume every prefix length is
    recoverable when its visible representations are indistinguishable.
12. Architecture comparisons must approximately match parameter/compute budgets. Equal FFN
    width is not fair when one design contains three large matrices and the other contains two.
13. A parameter reduction is not automatically a proportional FLOP reduction. For weight tying,
    distinguish removed parameter/optimizer traffic from the unchanged vocabulary projection.
14. Use equal steps/tokens **and hold other settings fixed** to isolate whether a modification learns
    more per update; use equal wall time to measure the tuned system's combined benefit under a budget.
15. Do not label the whole Transformer quadratic in context length: only score/value mixing is `T²`;
    token-wise projections, FFNs, and the LM head remain linear in `T`.

## Review debt

- Owner correction/confirmation of this draft.
- Recheck: scaling all coordinates by `c>0` is approximately cancelled by RMSNorm.
- Recheck: shifting Q and K positions together preserves RoPE's direct relative-position contribution.
- Owner confirmation of the generation/decoding section; greedy determinism was already
  correctly predicted in-session.
- RMSNorm ablation transfer check passed 2026-08-14: persistent `10^5` raw gradients clipped
  to 1 plus oscillating validation loss were correctly diagnosed as unstable despite no NaNs.
- Pre/post-norm warmup transfer passed 2026-08-14: longer warmup was correctly identified as
  unable to guarantee stability once the unchanged unsafe peak LR is reached.
- NoPE review debt: the owner understood nested-prefix order inference but initially expected
  a small difference for identical repeated tokens; review the exact softmax-average symmetry.
- SwiGLU transfer passed 2026-08-16: a near-zero gate was correctly interpreted as
  context-dependent suppression of the corresponding large content activation.
