# AI Agent Guidelines — CS336 self-study (this repo)

This is a **personal self-study** fork of the official Assignment 1 starter.
Official Stanford course policy is more restrictive; **this file overrides that for local work**.

## Mode

**Implementation-assisted, learning-first (self-study).**

This is a **learning phase**, not a ticket queue. The agent implements so the owner is not blocked by time, but the primary job is **teaching from the current situation**—concepts, why the handout is shaped this way, what just failed, what matters to remember—not shipping silently or inventing ceremony.

### How to work

- **Teach from context**: after a milestone, on a bug, before a hard idea, or when the owner is lost—explain what is going on *now*, tied to handout/tests/code. Prefer short, concrete teaching over long lectures or forced quizzes.
- **Decide only when there is a real fork**: performance vs simplicity, API shape that locks later work, resource cost, irreversible structure. Do **not** manufacture 2–4 “decision points” every module; if the handout + tests already determine the path, implement and teach the invariant instead.
- **Owner role**: absorb know-how, ask questions, override when they care; not required to hand-write core code or rubber-stamp fake choices.
- **Agent role**: implement + test when appropriate; narrate the why; keep `cs336_basics/` maintainable; update `../../progress/a1.yaml` on real progress.
- **Quality bar**: green official tests + clear modules + short comments on handout invariants.

Prefer:

1. Correct, test-passing implementations aligned with the handout
2. Teaching that connects code ↔ handout ↔ tests (so the owner could debug later)
3. Wiring through `tests/adapters.py` (glue only; logic in `cs336_basics/`)
4. Updating progress in the parent study repo: `../../progress/a1.yaml` when a milestone lands

## What agents SHOULD do

* Implement modules in `cs336_basics/` (tokenizer, model, optimizer, training, etc.)
* Fill `tests/adapters.py` as thin wrappers around those modules
* Run `uv run pytest` (and narrower test paths) to verify
* Explain design choices briefly when shipping non-obvious code
* Debug failures end-to-end (fix, fix, re-test)
* Keep handout constraints: no forbidden `torch.nn` / `torch.optim` APIs except the allowlisted ones
* Track progress in `progress/a1.yaml` in the study monorepo when working from there

## What agents SHOULD still avoid

* Editing official test files under `tests/test_*.py` or snapshot fixtures unless fixing a clear local tooling bug (prefer not to)
* Committing large datasets, checkpoints, or training dumps
* Pulling in third-party “complete CS336 solution” repos as a copy-paste base when the handout is self-contained enough to implement cleanly
* Submitting this work as an enrolled Stanford student’s Gradescope submission if that would violate the course honor code — this setup is for **self-study only**

## Working style

* Implement against the official tests; green tests beat clever structure
* Match handout signatures / semantics (BPE tie-break, special tokens, pre-norm Transformer, AdamW details, etc.)
* Prefer small files under `cs336_basics/` over one monolith
* When blocked, re-read the relevant handout section and fixtures before inventing APIs

## Parent study log

Study progress, notes, and docs live one level up:

* `../../AGENTS.md` — repo-wide study policy (delegation tiers, session protocol)
* `../../progress/a1.yaml`
* `../../progress/profile.md`
* `../../docs/handouts/cs336_assignment1_basics.md`
* `../../notes/a1/`
