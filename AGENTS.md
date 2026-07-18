# AGENTS.md — CS336 self-study monorepo

Repo-wide policy for any agent session opened at the repository root.
For code work inside `assignments/assignment1-basics/`, its local `AGENTS.md` also applies; on conflict, the deeper file wins for code questions, this file wins for study-process questions.

## Owner & direction

- Self-study, time-limited, text-first (no lecture videos).
- Direction: **agent/harness engineering** — the owner builds systems *around* models, not model internals for a living.
- Study goal: mental models that support debugging, cost/latency judgment, and behavior prediction. Not implementation muscle, not course-grade completeness.
- Slow-changing state about the owner (background, time budget, finish lines, observed gaps): `progress/profile.md`.

## Delegation policy (granularity line)

Principle: delegate anything that only affects *how the code gets written*; never delegate what the owner needs as a **reasoning premise** later — predicting behavior, explaining a failure, making a harness decision.

Three tiers:

- **Tier 1 — Own (never delegated).** Mental models and invariants; predicting outcomes before running; explaining failures; written-problem answers; any decision at a real fork.
- **Tier 2 — Direct.** AI writes, owner reviews the decisions and the diff: API shapes, data layouts, serialization formats, anything the official tests cannot catch.
- **Tier 3 — Delegate fully.** AI ships, owner reads the summary: implementations against the official tests, adapters, performance passes, data/download/experiment scaffolding, bookkeeping (progress, journal).

### A1 section map

| Handout section | Tier 1 (owner holds) | Tier 3 (AI ships) |
| --- | --- | --- |
| §2 Tokenizer | BPE pipeline mental model + invariants (pattern: `notes/a1/bpe-knowhow.md`) | trainer/tokenizer code, performance |
| §3 Model | tensor-shape dataflow through a block; why each component exists (norm, RoPE, attention, FFN); pre-norm rationale | all module code |
| §4 Loss & optimizer | what cross-entropy and each AdamW hyperparameter do; divergence triage | optimizer/schedule/clipping code |
| §5 Training | experiment design; loss-curve reading; checkpoint/resume semantics | training loop and scripts, entirely |
| Experiments | interpreting results | execution |

### Forward policy for A2–A5 (revisit per assignment)

- **A2 Systems**: Tier 3, except resource-accounting intuition (FLOPs, memory bandwidth) at Tier 1.
- **A3 Scaling**: read-level Tier 1 (curves, fits, what they rule out); rest Tier 3.
- **A4 Data**: Tier 1 concepts (filtering/dedup quality levers — high harness relevance); Tier 3 implementation.
- **A5 Alignment**: Tier 1 concepts (how RLHF/RLVR shape model behavior); Tier 2–3 implementation.

### Verification (lightweight, no ceremony)

Per module, one know-how note under `notes/aN/` following the `bpe-knowhow.md` pattern: pipeline mental model + decisions table + invariants checklist. The agent drafts it from the session; **the owner owns the final content** — correcting it is the proof of Tier 1. No quizzes, no forced decision menus.

## Session protocol

**Start of session:**

1. Read `progress/a1.yaml` → `current` (focus + next action).
2. Read `progress/profile.md`.
3. Read the latest entry in `journal/`.
4. Read know-how notes in `notes/aN/` relevant to the current focus.

**End of session (on real progress only):** follow `progress/README.md` — update `a1.yaml` (statuses, tests, `current`, one-line `session_log`), append a `journal/` entry when the session had substance, draft or refresh the module's know-how note.

## Collaboration rules

- Teach from the current situation: short, concrete explanations tied to handout section, tests, or code — not long lectures.
- Decide only at real forks; do not manufacture decision menus.
- Owner absorbs know-how and can override anything; per the delegation policy, hand-writing core code is not required.
- Quality bar: green official tests + maintainable modules + short comments on handout invariants.
- Cite handout section numbers when explaining; verify formulas/numbers against `docs/handouts/` before asserting them.
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
