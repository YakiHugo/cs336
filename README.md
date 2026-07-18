# CS336 Study Log

This repository tracks my self-study of Stanford CS336, Language Modeling from Scratch, Spring 2026.

Current starting point: text-first study (handouts + lecture extracts) and Assignment 1. Videos are optional.

## Progress

- **Tracker**: [progress/a1.yaml](./progress/a1.yaml) — checklist, current focus, test status
- **Writeup draft**: [notes/a1/writeup.md](./notes/a1/writeup.md) — written problem answers

## Text-first path (skip videos)

Start here:

1. [docs/text-first-study.md](./docs/text-first-study.md) — how to study without YouTube
2. [docs/handouts/cs336_assignment1_basics.md](./docs/handouts/cs336_assignment1_basics.md) — A1 handout (searchable)
3. [docs/lectures/](./docs/lectures/) — all Spring 2026 lectures as markdown

Refresh official sources + re-extract:

```sh
./scripts/sync_course_docs.sh
```

## Structure

- `AGENTS.md`: repo-wide agent policy — delegation tiers, session protocol, conventions.
- `assignments/assignment1-basics/`: local copy of the official Assignment 1 starter repo.
- `docs/resources.md`: official course links, schedule, and data sources.
- `docs/lectures/`: extracted lecture text (tracked).
- `docs/handouts/`: extracted assignment handouts (tracked).
- `notes/`: `notes/lectures/` lecture notes; `notes/aN/` per-assignment writeups and know-how.
- `journal/`: dated study log entries.
- `progress/`: live state — `a1.yaml` tracker + `profile.md` learner profile.
- `materials/raw/`: local-only PDFs/`.py` sources; ignored by Git.
- `scripts/`: setup/download helpers (`sync_course_docs.sh`, `download_a1_data.sh`).

## Quickstart

Assignment 1 is a standalone `uv` project:

```sh
cd assignments/assignment1-basics
uv sync
uv run pytest
```

The initial test suite is expected to fail with `NotImplementedError` until the assignment adapters are connected to student-written implementations.

To fetch the Assignment 1 TinyStories and OpenWebText sample data:

```sh
./scripts/download_a1_data.sh
```

## Study Rules

Self-study, **learning-first + implementation-assisted**: AI may write code and run tests so time is not the bottleneck, but should teach from the current situation (concepts, failures, milestones). Real technical decisions only when there is a genuine fork—not forced decision menus. Repo policy: [`AGENTS.md`](./AGENTS.md) (delegation tiers, session protocol); code-level rules: `assignments/assignment1-basics/AGENTS.md`. Respect handout API allowlists; this is not an enrolled Gradescope submission workflow.

## Sources

The authoritative source for this project is the official CS336 site and the `stanford-cs336` GitHub organization. See `docs/resources.md` for the current source list.

