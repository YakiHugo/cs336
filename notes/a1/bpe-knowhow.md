# BPE module — know-how & decisions (retroactive)

Code: `assignments/assignment1-basics/cs336_basics/bpe.py`  
Adapters: `get_tokenizer`, `run_train_bpe`  
Tests: `test_train_bpe.py`, `test_tokenizer.py` (green)

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
| 3 | Train speed strategy | Correct full recount of pair counts each merge | Incremental pair index (faster, more code) | Unit tests pass (~1s corpus.en / ~5s special-token fixture). Full TinyStories 10k vocab will likely need incremental + multiprocessing pretok |
| 4 | Encode merge algorithm | Repeatedly merge **lowest-rank** pair present in the word (GPT-2 style) | Walk the merge list in order applying each once | Both aim to match training order; GPT-2 style matches tiktoken tests |
| 5 | Overlapping specials | Regex longest-first (`<|eot|><|eot|>` before `<|eot|>`) | First-listed wins | Required by `test_overlapping_special_tokens` |
| 6 | Serialization | Optional hex merges + JSON vocab lists via `save_vocab_and_merges` | GPT-2 printable-byte encoding | Independent of unit tests; used only if we dump trained tokenizers for experiments |

## Invariants you should remember (debug checklist)

1. **No merge across pretokens** — spaces often *start* a pretok (`" the"`), so space is part of the token, not a separator you strip.
2. **Specials don’t enter pair counts** during training — split them out; only add their bytes to vocab.
3. **Tie-break** — max frequency, then lexicographically greater pair (`max` on `(count, pair)`).
4. **Decode** — never assume IDs form valid UTF-8; always `errors='replace'`.
5. **Adapters stay thin** — logic lives in `cs336_basics/`, not `tests/adapters.py`.

## Where to edit later

| Goal | Touch |
| --- | --- |
| Faster full-corpus training | `train_bpe`: incremental pair counts; pretok via `multiprocessing` + `pretokenization_example.find_chunk_boundaries` |
| Different special handling | `Tokenizer.__init__` / `encode` special regex |
| Dump/load for experiments | `save_vocab_and_merges` / `Tokenizer.from_files` |
| Match a different reference merge order | `_count_pairs` + tie-break + merge application |

## Open owner votes (still useful)

Even though code is green, you can lock preferences for the next refactor:

1. Keep one file vs split train/tokenizer?
2. Optimize train now, or only when we hit TinyStories experiment?
3. Want a tiny `scripts/train_tokenizer_debug.py` for local experiments?
