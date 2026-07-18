# Text-First Study Path (no videos)

This repo is set up so you can study CS336 from **handouts + lecture text/slides** without watching the YouTube recordings.

Official lectures are either:

- **Executable lectures** (`.py`): narrative + code in [stanford-cs336/lectures](https://github.com/stanford-cs336/lectures). Online viewer: https://cs336.stanford.edu/lectures/?trace=lecture_01
- **Slide PDFs** (`.pdf`): guest-style or slide decks in the same repo

## Where things live

| Kind | Git-tracked (searchable) | Local originals (gitignored) |
| --- | --- | --- |
| Lecture text | `docs/lectures/lecture_XX.md` | `materials/raw/lectures/` (`.py` / `.pdf`) |
| Assignment handouts | `docs/handouts/*.md` | `materials/raw/assignments/*.pdf` |
| Catalog + links | `docs/resources.md` | — |

Refresh local binaries + re-extract text:

```sh
./scripts/sync_course_docs.sh
```

## Recommended order for Assignment 1 (text-only)

1. **Handout (primary)**: `docs/handouts/cs336_assignment1_basics.md`  
   Canonical PDF: `materials/raw/assignments/cs336_assignment1_basics.pdf` (also under `assignments/assignment1-basics/`)
2. **Lecture 01** (tokenization): `docs/lectures/lecture_01.md`
3. **Lecture 02** (PyTorch / resource accounting): `docs/lectures/lecture_02.md` — useful, not strictly required to start coding
4. **Lecture 03** (architectures / hyperparameters): `docs/lectures/lecture_03.md` — aligns with the model section of A1
5. Implement against tests in `assignments/assignment1-basics/tests/`; keep `AGENTS.md` AI policy in mind

Lectures 04–05 are still useful background (MoE, GPUs) but are closer to A2 territory.

## Schedule map (Spring 2026)

| # | Topic | Local text | A1 relevance |
| --- | --- | --- | --- |
| 1 | Overview, tokenization | `docs/lectures/lecture_01.md` | High — tokenizer |
| 2 | PyTorch, FLOPs, memory | `docs/lectures/lecture_02.md` | Medium — train/debug mindset |
| 3 | Architectures, hyperparameters | `docs/lectures/lecture_03.md` | High — Transformer LM |
| 4 | Attention alternatives, MoE | `docs/lectures/lecture_04.md` | Low for A1 |
| 5 | GPUs, TPUs | `docs/lectures/lecture_05.md` | Low for A1 |
| 6 | Kernels, Triton | `docs/lectures/lecture_06.md` | A2 |
| 7–8 | Parallelism | `lecture_07.md`, `lecture_08.md` | A2 |
| 9, 11 | Scaling laws | `lecture_09.md`, `lecture_11.md` | A3 |
| 10 | Inference | `lecture_10.md` | Later |
| 12 | Evaluation | `lecture_12.md` | Later |
| 13–14 | Data | `lecture_13.md`, `lecture_14.md` | A4 |
| 15–16 | Mid/post-training, RLVR | `lecture_15.md`, `lecture_16.md` | A5 |
| 17 | Multimodality | `lecture_17.md` | Optional |

Source schedule: https://cs336.stanford.edu/#schedule

## Quality notes on extractions

- **`.py` lectures** → markdown is extracted from `text("...")` calls. Code demos and images are only lightly noted; for full interactive code steps, open the web trace or the raw `.py` under `materials/raw/lectures/`.
- **`.pdf` lectures / handouts** → text extracted with pypdf. Layout is imperfect (especially multi-column slides); prefer the PDF when a formula or table looks garbled.

## Upstream

- Course site: https://cs336.stanford.edu/
- Lectures repo: https://github.com/stanford-cs336/lectures
- A1 code: https://github.com/stanford-cs336/assignment1-basics
