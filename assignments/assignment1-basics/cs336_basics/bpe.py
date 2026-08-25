"""Byte-level BPE training and tokenization.

Handout invariants (Assignment 1 §2):
- Initial vocab is all 256 bytes; special tokens are added as full vocab entries.
- Pre-tokenize with the GPT-2-style regex; never merge across pre-token boundaries.
- Special tokens are hard split boundaries during training (excluded from pair counts).
- On pair-count ties, merge the lexicographically greater pair (`max` on bytes tuples).
- Encode applies merges in creation order (rank); decode uses UTF-8 with errors='replace'.

Performance notes (same semantics, scaled to multi-GB corpora):
- Pre-token counting runs in parallel over special-token-aligned byte ranges
  (`find_chunk_boundaries` + a spawn-context process pool) for large files.
- The merge loop keeps pair counts incrementally (pair -> containing-words
  index) instead of recounting every pair after every merge, and picks the
  best pair from a lazy heap whose order matches `max` on (count, pair).
"""

from __future__ import annotations

import heapq
import json
import multiprocessing
import os
from collections import Counter
from collections.abc import Iterable, Iterator

import regex as re

from .pretokenization_example import find_chunk_boundaries

# GPT-2 pretokenizer (tiktoken PR form from the handout).
PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Type aliases for readability.
Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]

# Shared one-byte token objects: words are tuples of these singletons, which
# avoids a `bytes([b])` allocation per byte and keeps word hashing cheap.
_BYTE_TOKENS: tuple[bytes, ...] = tuple(bytes([i]) for i in range(256))

# Below this per-worker slice size, process-pool overhead is not worth it.
_PARALLEL_MIN_CHUNK_BYTES = 4 * 1024 * 1024


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[str]:
    """Count pre-token frequency; specials split the corpus and are not counted.

    Keyed by the raw pre-token string (converted to `Word` once afterwards).
    `findall` and `Counter.update` both loop in C, which matters at GB scale.
    """
    counts: Counter[str] = Counter()
    if special_tokens:
        # Longer specials first so overlapping delimiters split correctly if ever needed.
        delim = "|".join(re.escape(st) for st in sorted(special_tokens, key=len, reverse=True))
        chunks = re.split(delim, text)
    else:
        chunks = [text]

    for chunk in chunks:
        if not chunk:
            continue
        counts.update(PRETOKEN_PATTERN.findall(chunk))
    return counts


def _count_pretokens_chunk(args: tuple[str, int, int, list[str]]) -> Counter[str]:
    """Worker: count pre-tokens in the byte range [start, end) of the corpus.

    Top-level function with picklable args/return so the spawn start method
    (macOS default) can run it; importing this module never spawns processes.
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    text = raw.decode("utf-8")
    # Mirror the universal-newline translation of the old single text-mode read.
    # Boundaries sit at special-token starts (pure ASCII), so a "\r\n" pair is
    # never split across chunks unless a special token itself starts with "\n".
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _count_pretokens(text, special_tokens)


def _count_pretokens_file(
    input_path: str | os.PathLike, special_tokens: list[str]
) -> Counter[str]:
    """Count pre-tokens, parallel over special-token-aligned chunks when worthwhile."""
    file_size = os.path.getsize(input_path)
    num_workers = min(os.cpu_count() or 4, file_size // _PARALLEL_MIN_CHUNK_BYTES + 1)
    if special_tokens and num_workers >= 2:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_workers, special_tokens[0].encode("utf-8"))
        ranges = list(zip(boundaries[:-1], boundaries[1:]))
        if len(ranges) >= 2:
            # Every boundary is a special-token start, so splitting each chunk
            # on the same specials partitions the corpus exactly. Merging the
            # per-chunk Counters is additive, hence deterministic.
            tasks = [(str(input_path), start, end, special_tokens) for start, end in ranges]
            counts: Counter[str] = Counter()
            with multiprocessing.get_context("spawn").Pool(processes=len(tasks)) as pool:
                for partial in pool.imap(_count_pretokens_chunk, tasks):
                    counts.update(partial)
            return counts

    with open(input_path, encoding="utf-8") as f:
        return _count_pretokens(f.read(), special_tokens)


def _words_from_counts(counts: Counter[str]) -> dict[Word, int]:
    """Convert counted pre-token strings to single-byte-tuple words."""
    return {
        tuple(_BYTE_TOKENS[b] for b in token.encode("utf-8")): freq
        for token, freq in counts.items()
    }


def _index_pairs(word_freqs: dict[Word, int]) -> tuple[dict[Pair, int], dict[Pair, set[Word]]]:
    """One-pass build of pair counts plus the pair -> containing-words index."""
    pair_counts: dict[Pair, int] = {}
    pair_words: dict[Pair, set[Word]] = {}
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            words = pair_words.get(pair)
            if words is None:
                pair_words[pair] = {word}
            else:
                words.add(word)
    return pair_counts, pair_words


class _ReversePairOrder:
    """Invert pair ordering so a min-heap yields the lexicographically GREATEST
    pair among equal counts, matching `max(counts, key=(count, pair))` exactly."""

    __slots__ = ("pair",)

    def __init__(self, pair: Pair) -> None:
        self.pair = pair

    def __lt__(self, other: _ReversePairOrder) -> bool:
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReversePairOrder) and self.pair == other.pair


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

    word_freqs = _words_from_counts(_count_pretokens_file(input_path, special_tokens))

    # Vocab layout: 256 base bytes, then specials, then learned merges.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for special in special_tokens:
        vocab[next_id] = special.encode("utf-8")
        next_id += 1

    merges: list[Pair] = []
    num_merges = vocab_size - len(vocab)
    pair_counts, pair_words = _index_pairs(word_freqs)

    # Lazy max-heap over (count, pair): entries go stale as counts change and
    # are skipped on inspection. The pop order equals `max` on (count, pair):
    # highest count, ties broken by the lexicographically greater pair.
    heap: list[tuple[int, _ReversePairOrder]] = [
        (-count, _ReversePairOrder(pair)) for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    for _ in range(num_merges):
        if not pair_counts:
            break
        # Drop stale heap entries until the top is a live (pair, count) maximum.
        best_pair: Pair | None = None
        while heap:
            neg_count, rev = heap[0]
            if pair_counts.get(rev.pair) == -neg_count:
                best_pair = rev.pair
                heapq.heappop(heap)
                break
            heapq.heappop(heap)
        if best_pair is None or pair_counts[best_pair] <= 0:
            break

        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[next_id] = new_token
        next_id += 1

        # Incremental update: only words containing best_pair change, so only
        # their old pairs lose counts and their merged forms gain counts.
        merged_freqs: dict[Word, int] = {}
        for word in pair_words.pop(best_pair):
            freq = word_freqs.pop(word)
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                remaining = pair_counts[pair] - freq
                if remaining == 0:
                    del pair_counts[pair]
                    pair_words.pop(pair, None)
                else:
                    pair_counts[pair] = remaining
                    words = pair_words.get(pair)
                    if words is not None:
                        words.discard(word)
                    heapq.heappush(heap, (-remaining, _ReversePairOrder(pair)))
            merged = _merge_word(word, best_pair, new_token)
            merged_freqs[merged] = merged_freqs.get(merged, 0) + freq

        for merged, freq in merged_freqs.items():
            word_freqs[merged] = word_freqs.get(merged, 0) + freq
            for i in range(len(merged) - 1):
                pair = (merged[i], merged[i + 1])
                count = pair_counts.get(pair, 0) + freq
                pair_counts[pair] = count
                words = pair_words.get(pair)
                if words is None:
                    pair_words[pair] = {merged}
                else:
                    words.add(merged)
                heapq.heappush(heap, (-count, _ReversePairOrder(pair)))

        # Compaction guard: stale entries would otherwise grow the heap without bound.
        if len(heap) > 4 * len(pair_counts) + 1024:
            heap = [
                (-count, _ReversePairOrder(pair)) for pair, count in pair_counts.items()
            ]
            heapq.heapify(heap)

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
