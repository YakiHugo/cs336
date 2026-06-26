# CS336 Study Log

This repository tracks my self-study of Stanford CS336, Language Modeling from Scratch, Spring 2026.

Current starting point: I have watched the first Spring 2026 YouTube lecture, "Overview, tokenization", and am beginning Assignment 1.

## Structure

- `assignments/assignment1-basics/`: local copy of the official Assignment 1 starter repo.
- `docs/resources.md`: official course links, assignment repos, lecture links, and data sources.
- `notes/`: lecture notes and reading notes.
- `journal/`: dated study log entries.
- `materials/raw/`: local-only downloaded course materials; ignored by Git.
- `scripts/`: small setup/download helpers.

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

The official Assignment 1 starter includes `AGENTS.md`, which frames AI assistants as teaching assistants rather than solution generators. For this repository, AI help should stay within explanations, debugging guidance, tests/sanity checks, review, and environment setup. Core assignment implementations should be written by me.

## Sources

The authoritative source for this project is the official CS336 site and the `stanford-cs336` GitHub organization. See `docs/resources.md` for the current source list.

