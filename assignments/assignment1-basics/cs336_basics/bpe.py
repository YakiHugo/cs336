"""Byte-level BPE training and tokenization.

Handout invariants (Assignment 1 §2):
- Initial vocab is all 256 bytes; special tokens are added as full vocab entries.
- Pre-tokenize with the GPT-2-style regex; never merge across pre-token boundaries.
- Special tokens are hard split boundaries during training (excluded from pair counts).
- On pair-count ties, merge the lexicographically greater pair (`max` on bytes tuples).
- Encode applies merges in creation order (rank); decode uses UTF-8 with errors='replace'.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import BinaryIO

import regex as re

# GPT-2 pretokenizer (tiktoken PR form from the handout).
PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Type aliases for readability.
Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


def _bytes_to_word(token: str) -> Word:
    raw = token.encode("utf-8")
    return tuple(bytes([b]) for b in raw)


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[Word]:
    """Count pre-token frequency; specials split the corpus and are not counted."""
    counts: Counter[Word] = Counter()
    if special_tokens:
        # Longer specials first so overlapping delimiters split correctly if ever needed.
        delim = "|".join(re.escape(st) for st in sorted(special_tokens, key=len, reverse=True))
        chunks = re.split(delim, text)
    else:
        chunks = [text]

    for chunk in chunks:
        if not chunk:
            continue
        for match in PRETOKEN_PATTERN.finditer(chunk):
            counts[_bytes_to_word(match.group(0))] += 1
    return counts


def _count_pairs(word_freqs: dict[Word, int]) -> Counter[Pair]:
    pair_counts: Counter[Pair] = Counter()
    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq
    return pair_counts


def _merge_word(word: Word, pair: Pair, new_token: bytes) -> Word:
    """Replace every non-overlapping occurrence of `pair` in `word` with `new_token`."""
    left, right = pair
    out: list[bytes] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == left and word[i + 1] == right:
            out.append(new_token)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train a byte-level BPE tokenizer.

    Returns
    -------
    vocab : dict[int, bytes]
        Token ID → token bytes.
    merges : list[tuple[bytes, bytes]]
        Merge rules in creation order.
    """
    if vocab_size < 256 + len(special_tokens):
        raise ValueError(
            f"vocab_size={vocab_size} too small for 256 bytes + {len(special_tokens)} specials"
        )

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    word_freqs: dict[Word, int] = dict(_count_pretokens(text, special_tokens))

    # Vocab layout: 256 base bytes, then specials, then learned merges.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for special in special_tokens:
        vocab[next_id] = special.encode("utf-8")
        next_id += 1

    merges: list[Pair] = []
    num_merges = vocab_size - len(vocab)
    pair_counts = _count_pairs(word_freqs)

    for _ in range(num_merges):
        if not pair_counts:
            break
        # Highest frequency; tie-break with lexicographically greater pair.
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        if pair_counts[best_pair] <= 0:
            break

        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[next_id] = new_token
        next_id += 1

        # Apply merge to all words; recompute pair counts for affected words only.
        # Full recount of pairs is simpler and still fast enough for the unit tests;
        # we update word_freqs in place then rebuild pair_counts for correctness.
        new_word_freqs: dict[Word, int] = {}
        for word, freq in word_freqs.items():
            merged = _merge_word(word, best_pair, new_token)
            new_word_freqs[merged] = new_word_freqs.get(merged, 0) + freq
        word_freqs = new_word_freqs
        pair_counts = _count_pairs(word_freqs)

    return vocab, merges


class Tokenizer:
    """BPE tokenizer: encode text → IDs, decode IDs → text."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[Pair] = list(merges)
        self.special_tokens: list[str] = list(special_tokens or [])

        # Append any specials missing from vocab (handout / tests).
        existing = set(self.vocab.values())
        for special in self.special_tokens:
            encoded = special.encode("utf-8")
            if encoded not in existing:
                new_id = max(self.vocab.keys(), default=-1) + 1
                self.vocab[new_id] = encoded
                existing.add(encoded)

        self._token_to_id: dict[bytes, int] = {tok: idx for idx, tok in self.vocab.items()}
        # Lower rank = earlier merge = higher priority when encoding.
        self._bpe_ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}

        if self.special_tokens:
            # Longest-first so overlapping specials match greedily.
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(st) for st in ordered)
            self._special_re = re.compile(f"({pattern})")
            self._special_set = set(self.special_tokens)
        else:
            self._special_re = None
            self._special_set = set()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load vocab/merges written by `save_vocab_and_merges` (raw JSON + text)."""
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        # JSON keys are strings; values are lists of ints (bytes).
        vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}

        merges: list[Pair] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                # Each merge line: two space-separated hex-encoded byte strings.
                left_hex, right_hex = line.split(" ")
                merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if self._special_re is None:
            return self._encode_ordinary(text)

        ids: list[int] = []
        # re.split with capturing group keeps specials in the piece list.
        for piece in self._special_re.split(text):
            if not piece:
                continue
            if piece in self._special_set:
                ids.append(self._token_to_id[piece.encode("utf-8")])
            else:
                ids.extend(self._encode_ordinary(piece))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in PRETOKEN_PATTERN.finditer(text):
            pretok = match.group(0).encode("utf-8")
            for piece in self._bpe(pretok):
                ids.append(self._token_to_id[piece])
        return ids

    def _bpe(self, token: bytes) -> list[bytes]:
        """Apply merges to one pre-token's UTF-8 bytes."""
        if not token:
            return []
        word: list[bytes] = [bytes([b]) for b in token]
        if len(word) == 1:
            return word

        ranks = self._bpe_ranks
        while True:
            if len(word) < 2:
                break
            # Best (lowest-rank) adjacent pair present in the merge table.
            best_rank = None
            best_pair: Pair | None = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            word = list(_merge_word(tuple(word), best_pair, best_pair[0] + best_pair[1]))
        return word


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[Pair],
    vocab_filepath: str | os.PathLike,
    merges_filepath: str | os.PathLike,
) -> None:
    """Serialize vocab/merges for later `Tokenizer.from_files` (hex merges)."""
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump({str(k): list(v) for k, v in vocab.items()}, f)
    with open(merges_filepath, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left.hex()} {right.hex()}\n")
