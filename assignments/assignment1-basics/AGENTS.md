# AI Agent Guidelines — CS336 self-study (this repo)

This is a **personal self-study** fork of the official Assignment 1 starter.
Official Stanford course policy is more restrictive; **this file overrides that for local work**.

## Scope

The repository-root `../../AGENTS.md` is the sole authority for the learning process, delegation tiers, AI-led teaching loop, and progress semantics. Follow its `progress/a1.yaml → current` handoff before starting code work. This file adds only Assignment 1 code and verification constraints; it does not redefine the pedagogy.

This is implementation-assisted self-study. The agent may implement all Tier 2/3 code, but must keep logic maintainable, connect tests and failures to the handout invariants selected by the root teaching loop, and update the parent tracker when a milestone lands.

Code priorities:

1. Correct, test-passing implementations aligned with the handout.
2. Logic in `cs336_basics/`; `tests/adapters.py` remains thin glue.
3. Short comments on non-obvious handout invariants and tensor shapes.
4. `../../progress/a1.yaml` updated from actual test evidence in the same session.

## What agents SHOULD do

* Implement modules in `cs336_basics/` (tokenizer, model, optimizer, training, etc.)
* Fill `tests/adapters.py` as thin wrappers around those modules
* Run `uv run pytest` (and narrower test paths) to verify
* Debug failures end-to-end (fix, fix, re-test)
* Keep handout constraints: no forbidden `torch.nn` / `torch.optim` APIs except the allowlisted ones
* Track progress in `progress/a1.yaml` in the study monorepo when working from there

## What agents SHOULD still avoid

* Editing official test files under `tests/test_*.py` or snapshot fixtures unless fixing a clear local tooling bug (prefer not to)
* Committing large datasets, checkpoints, or training dumps
* Pulling in third-party “complete CS336 solution” repos as a copy-paste base when the handout is self-contained enough to implement cleanly
* Submitting this work as an enrolled Stanford student’s Gradescope submission if that would violate the course honor code — this setup is for **self-study only**

## Code working style

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
