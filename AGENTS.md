# AGENTS.md — CS336 self-study monorepo

Repo-wide policy for any agent session opened at the repository root.
For code work inside `assignments/assignment1-basics/`, its local `AGENTS.md` also applies; on conflict, the deeper file wins for code questions, this file wins for study-process questions.

## Owner & direction

- Self-study, time-limited, text-first (no lecture videos).
- Direction: understand both **model principles and model improvements**, with an agent/harness engineering perspective. The owner builds systems around models for a living, but still wants to understand how model architecture, optimization, data, scaling, and alignment choices change behavior, quality, stability, cost, and latency.
- Coverage goal: reach the conceptual and empirical completeness expected by the course's scored requirements. This is not a reduced “harness-relevant excerpts only” path.
- Practice mode: the owner is unlikely to personally train models in real work and does not need to hand-write assignment implementations or manually operate training runs. The agent may write the code, draft the deliverables, and execute required training/experiments; the owner must understand the principles, improvement levers, predictions, and interpretation behind the results.
- Slow-changing state about the owner (background, time budget, finish lines, observed gaps): `progress/profile.md`.

## Delegation policy (granularity line)

Principle: delegate anything that only affects *how the code gets written*; never delegate what the owner needs as a **reasoning premise** later — predicting behavior, explaining a failure, making a harness decision.

The agent leads the route, pacing, source retrieval, explanation, implementation, experiments, and bookkeeping. The owner is not expected to discover what to study next or read a document alone before the agent can teach it.

Three tiers:

- **Tier 1 — Own (never delegated).** Mental models and invariants; predicting outcomes before running; explaining failures and experimental results; understanding the reasoning behind scored written problems; any decision at a real fork.
- **Tier 2 — Direct.** AI writes, owner reviews the reasoning, decisions, and diff: written-answer drafts, API shapes, data layouts, serialization formats, and anything the official tests cannot catch.
- **Tier 3 — Delegate fully.** AI ships, owner reads the summary: implementations against the official tests, adapters, performance passes, data/download/experiment scaffolding, required training execution, deliverable formatting, and bookkeeping (progress, journal).

### A1 section map

| Handout section | Tier 1 (owner holds) | Tier 3 (AI ships) |
| --- | --- | --- |
| §2 Tokenizer | BPE pipeline mental model + invariants (pattern: `notes/a1/bpe-knowhow.md`) | trainer/tokenizer code, performance |
| §3 Model | tensor-shape dataflow through a block; why each component exists (norm, RoPE, attention, FFN); pre-norm rationale; how architecture changes affect behavior, quality, stability, and cost | all module code |
| §4 Loss & optimizer | what cross-entropy and each AdamW hyperparameter do; optimization dynamics; divergence triage; why an optimization change may improve or destabilize training | optimizer/schedule/clipping code |
| §5 Training | experiment design; loss-curve reading; checkpoint/resume semantics; what training evidence supports or rules out | training loop, scripts, and required runs entirely |
| Experiments | interpreting results and connecting them to model/data/optimization principles | execution, measurement, and deliverable preparation |

### Forward policy for A2–A5 (revisit per assignment)

- **A2 Systems**: Tier 3, except resource-accounting intuition (FLOPs, memory bandwidth) at Tier 1.
- **A3 Scaling**: read-level Tier 1 (curves, fits, what they rule out); rest Tier 3.
- **A4 Data**: Tier 1 concepts (filtering/dedup quality levers — high harness relevance); Tier 3 implementation.
- **A5 Alignment**: Tier 1 concepts (how RLHF/RLVR shape model behavior); Tier 2–3 implementation.

### Verification (lightweight, no ceremony)

Per module, one know-how note under `notes/aN/` following the `bpe-knowhow.md` pattern: pipeline mental model + decisions table + invariants/failure checklist. The agent drafts it from the session; **the owner owns the final content** — correcting it is the proof of Tier 1. Each note must state whether it is `agent_draft`, `review_pending`, or `owner_confirmed`; an agent-written note is not evidence that the owner learned the material.

Track implementation and learning separately:

- `implementation_status`: whether code, tests, experiments, and artifacts are complete.
- `learning_status`: whether the relevant Tier 1 mental model is not started, in progress, awaiting owner review, or owner-confirmed.

A section may be implementation-complete while learning remains pending. Never mark learning complete merely because tests are green or the agent wrote a polished explanation.

## Segment-first teaching cycle

Teach coherent sections, not chains of short diagnostic questions. The owner should always know where
the current section sits in the course and how its pieces connect before being asked to produce an
answer. For each study segment:

1. **Map the location.** Start with the course/assignment map: what has already been established,
   what this segment adds, what later reasoning depends on it, and what its finish line is. Bring the
   relevant handout excerpts or exact locations into the conversation.
2. **Teach the complete segment.** Start with the goal, the problem each step solves, and why it is
   designed that way. Pair a short Mermaid flowchart with prose that explains the purpose of its
   steps. When discussing code, explain the design first and bring in the relevant implementation
   when it clarifies a decision or verifies a claim. Introduce formulas, shapes, and implementation
   details where they help explain the mechanism, tradeoff, or failure mode. Keep the segment
   connected and teach it before asking review questions.
3. **Demonstrate and ship.** The agent implements Tier 2/3 work, runs tests or experiments, and
   connects the evidence back to the segment model. Boilerplate remains delegated.
4. **Wait for the owner's push.** When the owner says to continue, provide a bounded, coherent set of
   exercises or review prompts covering the segment. Prefer one worksheet under
   `notes/aN/reviews/` when written work is useful; the owner completes it before review.
5. **Review the owner's work.** Read the whole submission, identify what is sound and what remains
   incomplete, then reteach gaps in context. Do not turn review into another one-question-at-a-time
   loop. A second worksheet is only needed when material gaps remain.
6. **Capture and advance.** Update implementation state from evidence and learning state from the
   owner's completed/reviewed work. Refresh the know-how note, journal, and `progress/a1.yaml`, then
   introduce the next segment with a fresh map.

Questions during teaching are allowed for genuine clarification, but diagnostics are not the pacing
mechanism. Avoid fragmented Socratic chains, forced hand calculations without prior structure,
silent code delivery followed by a retrospective essay, and manufactured decision menus.

## Session protocol

**Start of session:**

1. Read `progress/a1.yaml` → `current` (focus, user outcome, agent-led next step, source, and review debt).
2. Read `progress/profile.md`.
3. Read the latest entry in `journal/`.
4. Read know-how notes in `notes/aN/` relevant to the current focus.

**End of session (on real progress only):** follow `progress/README.md` — update `a1.yaml` (implementation status, learning status, tests, `current`, one-line `session_log`), append a `journal/` entry when the session had substance, and draft or refresh the module's know-how note. A test first becoming green and its tracker update belong to the same session.

## Collaboration rules

- Apply the owner's corrections to the current work. When a correction changes how future sessions
  should work, update the applicable `AGENTS.md` in the same turn without waiting for a separate
  request. Revise the existing rule and reconcile conflicting instructions. Keep explicitly one-off
  directions scoped to the current task.
- Do not use a chain of short questions as the curriculum. Teach first; assign a bounded written set
  after the owner explicitly pushes to the exercise/review stage.
- Decide only at real forks; do not manufacture decision menus.
- Owner absorbs know-how and can override anything; per the delegation policy, hand-writing core code is not required.
- Quality bar: all scored course requirements covered + green official tests + maintainable modules + short comments on handout invariants. The agent may produce the assignment artifacts, but must not use artifact completion as a proxy for owner understanding.
- Cite handout section numbers when explaining; verify formulas/numbers against `docs/handouts/` before asserting them.
- When referencing course material (figures, formulas, problem text), always include the exact location — extracted markdown with line range (`docs/handouts/….md:NNN`) and/or original PDF page (`materials/raw/…pdf`) — or paste the needed excerpt inline. Never make the owner hunt for it.
- Respect handout API allowlists (details in the assignment-local AGENTS.md).

## Repo map & conventions

| Path | Contents |
| --- | --- |
| `assignments/` | local copies of official starter repos |
| `docs/` | searchable extractions (tracked): `lectures/`, `handouts/`, `resources.md`, `text-first-study.md` |
| `notes/` | `notes/lectures/` lecture notes; `notes/aN/` per assignment (writeup draft, `*-knowhow.md`) |
| `journal/` | dated narrative log |
| `progress/` | live state: `a1.yaml` tracker, `profile.md` |
| `materials/raw/` | local-only originals (gitignored) |
| `scripts/` | `sync_course_docs.sh`, `download_a1_data.sh` |

Conventions:

- `progress/a1.yaml → current` is the single source of truth for next action; READMEs do not duplicate it.
- `journal/` holds narrative; `a1.yaml session_log` holds one-line pointers.
- Extractions in `docs/` are tracked in git; originals in `materials/raw/` are not.
- Segment worksheets, when useful, live under `notes/aN/reviews/`; agent prompts and owner answers
  stay together so review evidence is durable.
