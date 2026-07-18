# Progress tracking

| File | Purpose |
| --- | --- |
| [a1.yaml](./a1.yaml) | Assignment 1 checklist + current focus |
| [../notes/a1/writeup.md](../notes/a1/writeup.md) | Written problem answers (unicode*, experiments, …) |

## How we update

After each study session (you or the assistant):

1. Set `current.focus` and `current.next_action` in `a1.yaml`
2. Flip item `status`: `not_started` → `in_progress` → `done` (or `blocked`)
3. When a pytest file goes green, set matching `tests: passing` and `test_suites.*: passing`
4. Append a line under `session_log`

Quick test commands (from `assignments/assignment1-basics/`):

```sh
uv run pytest tests/test_train_bpe.py -q
uv run pytest tests/test_tokenizer.py -q
uv run pytest -q
```
