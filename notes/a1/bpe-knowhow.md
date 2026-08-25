# BPE module — know-how & decisions (retroactive)

Learning status: **review_pending** — agent draft; owner corrections/confirmation still required.

Code: `assignments/assignment1-basics/cs336_basics/bpe.py`  
Adapters: `get_tokenizer`, `run_train_bpe`  
Tests: `test_train_bpe.py`, `test_tokenizer.py` — 26 passed, 2 skipped (pre-existing linux-only rlimit skips)

This was implemented with too little owner input. Below is what *should* have been decided together, and what you can still override later.

## Pipeline (mental model)

```
text
  → split on special tokens (hard boundaries)
  → GPT-2 regex pretokenize (per chunk)
  → each pretok = sequence of UTF-8 bytes
  → repeatedly merge most frequent adjacent pair
  → vocab: id → bytes ; merges: ordered list of (bytes, bytes)

encode: pretok → apply merges by rank → ids
decode: ids → concat bytes → utf-8 (errors='replace')
```

## Decisions already baked in (you can still change)

| # | Decision | What we shipped | Alternative | Why it matters |
| --- | --- | --- | --- | --- |
| 1 | File layout | Single `bpe.py` (`train_bpe` + `Tokenizer`) | Split `bpe_train.py` / `tokenizer.py` | One file is easier to scan now; split later if model code crowds the package |
| 2 | Vocab ID layout | `0..255` bytes → specials → merges | Specials first (handout toy example order) | Tests only check *set* of keys/values for train; encode uses `bytes→id` map so order is flexible as long as consistent |
| 3 | Train speed strategy | Rewritten 2026-07-19: parallel pretok + C-level counting + incremental merge loop (see 7–9) | Full pair recount each merge (original impl) | `test_train_bpe_speed` was red (1.64s > 1.5s); the rewrite is bit-identical to the old impl and made full-corpus trains cheap (16.9s TS / 74min OWT) |
| 4 | Encode merge algorithm | Repeatedly merge **lowest-rank** pair present in the word (GPT-2 style) | Walk the merge list in order applying each once | Both aim to match training order; GPT-2 style matches tiktoken tests |
| 5 | Overlapping specials | Regex longest-first (`<|eot|><|eot|>` before `<|eot|>`) | First-listed wins | Required by `test_overlapping_special_tokens` |
| 6 | Serialization | Optional hex merges + JSON vocab lists via `save_vocab_and_merges` | GPT-2 printable-byte encoding | Independent of unit tests; used only if we dump trained tokenizers for experiments |
| 7 | Parallel pretok axis | File byte-ranges aligned to special-token boundaries (`find_chunk_boundaries` on `b"<|endoftext|>"`), per-chunk `Counter`s merged by addition | Fixed-size or line-count splits | Splitting anywhere else can change pretok counts; additive Counter merge is deterministic and order-independent |
| 8 | Incremental merge loop | pair→words index + lazy heap keyed `(-count, reverse-lex pair)` | Full recount per merge; eager heap updates | Exactly reproduces `max` on `(count, pair)` tie-break; the recount was O(corpus) per merge and dominated runtime |
| 9 | Correctness harness | Throwaway equivalence script, old vs new, 8 scenarios incl. `PYTHONHASHSEED` 0/42 | Trust unit tests alone | Bit-identical vocab/merges across hash seeds rules out ordering nondeterminism; speed tests only check wall time |

## Performance (verified 2026-07-19)

- TinyStories 10K (2.1 GB train): **16.9 s wall, ~1.1 GB peak RSS**; artifacts `data/ts10k_vocab.json` + `ts10k_merges.txt`.
- OpenWebText 32K (11.9 GB train): **4461.8 s (74.4 min), 11.76 GB peak RSS**; artifacts `data/owt32k_vocab.json` + `owt32k_merges.txt`.
- Unit tests: **26 passed, 2 skipped** (pre-existing linux-only rlimit skips); `test_train_bpe_speed` was red before the rewrite (1.64s > 1.5s).
- Post-rewrite bottleneck: GPT-2 regex pretokenization/counting (~0.23s of ~0.62s on the 5MB fixture); the merge loop is milliseconds.
- Encode throughput: ~1.41 MB/s single-threaded pure Python (`encode_iterable`).

## Invariants you should remember (debug checklist)

1. **No merge across pretokens** — spaces often *start* a pretok (`" the"`), so space is part of the token, not a separator you strip.
2. **Specials don’t enter pair counts** during training — split them out; only add their bytes to vocab.
3. **Tie-break** — max frequency, then lexicographically greater pair (`max` on `(count, pair)`).
4. **Decode** — never assume IDs form valid UTF-8; always `errors='replace'`.
5. **Adapters stay thin** — logic lives in `cs336_basics/`, not `tests/adapters.py`.
6. **Longest tokens are high-frequency *within-pretoken* strings** — cross-word pairs lose the frequency race, so no phrases; but the pretoken class sets the ceiling: punctuation runs are single pretokens under the GPT-2 regex, hence 64-byte separator tokens in OWT.
7. **Vocab is a fossil record of the training distribution** — off-distribution tokenization degrades to short tokens/bytes (−26% B/tok measured TS→OWT); rare/undertrained embeddings cause glitch tokens (vocab entry ≠ trained representation).
8. **Parallelize only along invariant-respecting boundaries** (special-token/document boundaries); inherently sequential stages (the merge loop) need index-based work elimination, not more cores.

## Where to edit later

| Goal | Touch |
| --- | --- |
| Even faster training | `train_bpe`: parallel pretok + incremental merge loop shipped 2026-07-19; next lever would be native code (Rust/C++) |
| Different special handling | `Tokenizer.__init__` / `encode` special regex |
| Dump/load for experiments | `save_vocab_and_merges` / `Tokenizer.from_files` |
| Match a different reference merge order | `_count_pairs` + tie-break + merge application |

## Open owner votes (still useful)

All resolved 2026-07-19:

1. Keep one file vs split train/tokenizer? → **Single `bpe.py` kept.**
2. Optimize train now, or only when we hit TinyStories experiment? → **Optimized when the experiment arrived** (`test_train_bpe_speed` red forced it).
3. Want a tiny `scripts/train_tokenizer_debug.py` for local experiments? → **Not needed** — the throwaway equivalence harness covered local verification.
