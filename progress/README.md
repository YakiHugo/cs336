# Progress tracking

| File | Purpose |
| --- | --- |
| [a1.yaml](./a1.yaml) | Assignment 1 checklist + current focus |
| [../notes/a1/writeup.md](../notes/a1/writeup.md) | Written problem answers (unicode*, experiments, …) |

## How we update

`../AGENTS.md` is the sole teaching protocol. This directory records state; it does not duplicate pedagogy.

After each substantive study session:

1. Set the `current` handoff: focus, user outcome, agent-led next step, exact source, implementation next step, and review debt.
2. Update `implementation_status`: `not_started` → `in_progress` → `done` (or `blocked`).
3. Update `learning_status`: `not_started` → `in_progress` → `review_pending` → `owner_confirmed` (`not_applicable` where appropriate).
4. When a test first goes green, update the matching item and `test_suites.*` in the same session.
5. Append one concise evidence-based line under `session_log`.

Implementation and learning are independent: green tests do not confirm learning, and an agent-drafted know-how note remains `review_pending`. `current` is the only source of truth for what happens next; journal entries record history and must not create a competing next-action queue.

Quick test commands (from `assignments/assignment1-basics/`):

```sh
uv run pytest tests/test_train_bpe.py -q
uv run pytest tests/test_tokenizer.py -q
uv run pytest -q
```
