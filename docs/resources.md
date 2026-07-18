# CS336 Resources

Official materials checked on 2026-07-17.

## Text-first entrypoint (recommended if skipping videos)

- **Study path**: [text-first-study.md](./text-first-study.md)
- **Lectures (extracted)**: [lectures/](./lectures/)
- **Handouts (extracted)**: [handouts/](./handouts/)
- **Refresh script**: `./scripts/sync_course_docs.sh`
- **Local binaries** (gitignored): `materials/raw/lectures/`, `materials/raw/assignments/`

## Course

- Course site: https://cs336.stanford.edu/
- YouTube playlist (optional): https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV
- Lecture materials repo: https://github.com/stanford-cs336/lectures
- Interactive lecture traces: https://cs336.stanford.edu/lectures/?trace=lecture_01
- Schedule: https://cs336.stanford.edu/#schedule

## Spring 2026 schedule (A1-focused slice)

| # | Date (2026) | Description | Local text |
| --- | --- | --- | --- |
| 1 | Mon Mar 30 | Overview, tokenization | [lecture_01.md](./lectures/lecture_01.md) — A1 out |
| 2 | Wed Apr 1 | PyTorch, resource accounting | [lecture_02.md](./lectures/lecture_02.md) |
| 3 | Mon Apr 6 | Architectures, hyperparameters | [lecture_03.md](./lectures/lecture_03.md) |
| 4 | Wed Apr 8 | Attention alternatives, MoE | [lecture_04.md](./lectures/lecture_04.md) |
| 5 | Mon Apr 13 | GPUs, TPUs | [lecture_05.md](./lectures/lecture_05.md) |
| 6 | Wed Apr 15 | Kernels, Triton | [lecture_06.md](./lectures/lecture_06.md) — A1 due / A2 out |

Full list through lecture 17 is in [lectures/README.md](./lectures/README.md).

## Assignments

| Assignment | Topic | Official code | Handout text | Handout PDF (local) |
| --- | --- | --- | --- | --- |
| 1 | Basics | https://github.com/stanford-cs336/assignment1-basics | [A1](./handouts/cs336_assignment1_basics.md) | `materials/raw/assignments/cs336_assignment1_basics.pdf` |
| 2 | Systems | https://github.com/stanford-cs336/assignment2-systems | [A2](./handouts/cs336_assignment2_systems.md) | `materials/raw/assignments/cs336_assignment2_systems.pdf` |
| 3 | Scaling | https://github.com/stanford-cs336/assignment3-scaling | [A3](./handouts/cs336_assignment3_scaling.md) | `materials/raw/assignments/cs336_assignment3_scaling.pdf` |
| 4 | Data | https://github.com/stanford-cs336/assignment4-data | [A4](./handouts/cs336_assignment4_data.md) | `materials/raw/assignments/cs336_assignment4_data.pdf` |
| 5 | Alignment / Reasoning RL | https://github.com/stanford-cs336/assignment5-alignment | [A5](./handouts/cs336_spring2026_assignment5_alignment.md) | `materials/raw/assignments/cs336_spring2026_assignment5_alignment.pdf` |

## Assignment 1 data

- TinyStories train: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
- TinyStories valid: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
- OWT sample train: https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
- OWT sample valid: https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

Use `scripts/download_a1_data.sh` to download into `assignments/assignment1-basics/data/`.

## GPU compute (self-study)

Listed on the course site (public pricing around Mar 2026): Modal, Lambda, RunPod, Nebius, Together. Prefer correctness on CPU first; use GPU for real training/benchmarks.
