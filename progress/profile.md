# Learner profile

Slow-changing state about the owner. The agent reads this at session start (see root `AGENTS.md`) and appends observations as they surface.

## Direction

- Understand model principles and the main ways models are improved across architecture, optimization, data, scaling, and alignment.
- Apply that understanding from an agent/harness engineering perspective: debugging, behavior prediction, and quality/cost/latency judgment around models.
- Does not expect to implement or train model internals personally as a normal part of work, but does want course-level understanding rather than a reduced overview.

## Constraints

- Time-limited; text-first (no lecture videos).
- Owner-facing teaching, worksheets, review prompts, and conclusions must be written in Chinese. Keep
  English only where needed for code identifiers, standard technical terms, and exact source paths; do
  not assume the owner can follow English instructional prose.
- The M5 Pro laptop is intentionally kept in low-power mode; training infrastructure should optimize within that constraint rather than asking to change the power policy.
- Delegation granularity per root `AGENTS.md`: implementation is the agent's job, Tier-1 reasoning is the owner's.
- Meet the course's scored requirements, but do not require the owner to hand-write assignment code/answers or personally operate training. The agent executes and drafts; the owner learns and reviews the underlying reasoning and evidence.
- Teaching must be segment-first: establish the larger course location and explain a complete connected
  section before requesting answers. After the owner pushes forward, use a bounded worksheet/file for
  concentrated owner work and review it as a whole; avoid one-short-question-at-a-time pacing.

## Current finish line

- A1: all scored deliverables covered, all official tests green, required training/experiment evidence produced by the agent, and the owner can explain the model principles and improvement levers and interpret the resulting loss curves. Hand-writing or personally running training is not part of the finish line. (Owner-set; change freely.)

## Observed strengths / gaps

Running log — agent appends with a date when something real surfaces; owner can edit or delete.

- 2026-07-17: Unicode concepts picked up quickly from text. BPE shipped silently at first and had to be redone as a learning pass — owner retains more when decisions and invariants are surfaced during the work, not after.
- 2026-07-25: A small diagnostic question must be followed by systematic teaching of the surrounding knowledge. The owner wants the agent to lead the route, source retrieval, explanation, implementation, and pacing—not substitute a quiz or send them away to discover the lesson alone.
- 2026-07-30: The target is full course-assessment coverage, including model principles and improvement methods—not only harness-relevant excerpts. The owner is unlikely to train models personally; AI should write and execute the assignments/training while the owner masters the reasoning and interprets the evidence.
- 2026-08-02: Loss-curve triage is becoming concrete: the owner immediately identified train-down/validation-up as overfitting and correctly localized failure to memorize one fixed minibatch to the training chain rather than generalization.
- 2026-08-02: The owner correctly predicted the qualitative learning-rate sweep regimes: `1e-5` would learn too slowly, a middle rate would behave normally, and `3e-2` would show large instability. The remaining distinction to consolidate is short-horizon optimization speed versus final-run generalization/stability.
- 2026-08-03: The owner immediately recognized that comparing batch 1 and batch 64 at equal update count gives batch 64 sixty-four times more training tokens. Controlled-experiment fairness and token accounting are becoming reliable instincts.
- 2026-08-11: The owner correctly predicted that removing RMSNorm would allow internal values to grow and make training unstable. The remaining precision to consolidate is distinguishing activation/gradient scale drift from literal floating-point overflow, and recognizing clipping/lower LR as compensation rather than restoration of the normalization invariant.
- 2026-08-14: The owner passed the RMSNorm transfer check: correctly explained that persistent extreme pre-clip gradients plus oscillating validation loss are unstable training even without NaNs. This closes the prior gap between finite execution, clipping protection, and genuine optimization stability.
- 2026-08-14: The owner correctly transferred residual-path reasoning to a 40-layer comparison, choosing pre-norm for stability, and asked for the missing definition of warmup. Next check is whether warmup only protects entry into a peak LR or changes the safety of that peak itself.
- 2026-08-14: The owner correctly concluded that longer warmup cannot guarantee stability at an unchanged unsafe peak LR. The answer needed only a precision correction: warmup changes early LR levels rather than globally reducing each step-to-step LR increment.
- 2026-08-16: The owner correctly predicted that a NoPE decoder can infer some ordering from differences in visible causal prefixes. The remaining boundary to check is when attention over indistinguishable repeated tokens cannot recover exact position or count.
- 2026-08-16: The identical-token NoPE boundary required correction: the owner expected a small positional difference, but softmax over repeated identical values removes cardinality exactly. Preserve this as review debt while moving on; the broader nested-prefix mechanism was understood.
- 2026-08-16: The owner immediately rejected an equal-width SwiGLU/SiLU comparison because the three-versus-two matrix parameter counts differ. Experimental-confound detection remains a strong and reliable skill.
- 2026-08-16: The owner correctly interpreted a near-zero SwiGLU gate as dynamically suppressing the corresponding content feature, completing the gating mental model.
- 2026-08-25: The owner explicitly rejected the accumulated one-short-question-at-a-time method as
  too fragmented. Future sessions should map the current segment in the larger architecture, teach
  its full structure/details first, then let the owner push to a coherent written worksheet that the
  agent reviews before advancing.
- 2026-08-29: The owner does not want to transcribe answers for material they already understand.
  Accept self-reported coverage as a reason to omit redundant writing, but do not treat it as independent
  confirmation; concentrate durable review work only on explicitly identified gaps.
- 2026-11-18: The owner reported forgetting much of A1, having never read the implementation, and
  wanting to revisit the original principles. The code-first walkthrough felt tiring. Start with the
  goal, the design motivation, and the problem each step solves. Pair a short Mermaid flowchart with
  the explanation and bring in code when it helps explain a decision or verify a claim. Treat the
  long-term delegation and review rules as open for discussion.
