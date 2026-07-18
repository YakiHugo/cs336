# Learner profile

Slow-changing state about the owner. The agent reads this at session start (see root `AGENTS.md`) and appends observations as they surface.

## Direction

- Agent/harness engineering: designs systems around models; does not implement model internals for a living.
- Studies CS336 for debugging intuition, cost/latency judgment, and behavior prediction.

## Constraints

- Time-limited; text-first (no lecture videos).
- Delegation granularity per root `AGENTS.md`: implementation is the agent's job, Tier-1 reasoning is the owner's.

## Current finish line

- A1: all unit tests green + one TinyStories training run + owner interprets the loss curve. (Owner-set; change freely.)

## Observed strengths / gaps

Running log — agent appends with a date when something real surfaces; owner can edit or delete.

- 2026-07-17: Unicode concepts picked up quickly from text. BPE shipped silently at first and had to be redone as a learning pass — owner retains more when decisions and invariants are surfaced during the work, not after.
