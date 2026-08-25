"""Stream and parallelize corpus tokenization into resumable raw token arrays.

Chunks begin at the document delimiter, which is already a hard tokenizer boundary.
Concatenating chunk encodings therefore exactly preserves full-corpus tokenization while
allowing multi-process encoding and bounded parent-process memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from cs336_basics.bpe import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries

_WORKER_TOKENIZER: Tokenizer | None = None


def _initialize_worker(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def _encode_chunk(task: tuple[int, str, int, int, str]) -> tuple[int, np.ndarray, int]:
    index, input_path, start, end, dtype = task
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("encoder worker was not initialized")
    with open(input_path, "rb") as handle:
        handle.seek(start)
        raw = handle.read(end - start)
    text = raw.decode("utf-8")
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    token_ids = np.asarray(_WORKER_TOKENIZER.encode(text), dtype=np.dtype(dtype))
    return index, token_ids, len(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _build_manifest(
    input_path: Path,
    output_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_token: str,
    dtype: str,
    boundaries: list[int],
) -> dict[str, Any]:
    stat = input_path.stat()
    return {
        "version": 1,
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "output_path": str(output_path.resolve()),
        "vocab_path": str(vocab_path.resolve()),
        "vocab_sha256": _sha256(vocab_path),
        "merges_path": str(merges_path.resolve()),
        "merges_sha256": _sha256(merges_path),
        "special_token": special_token,
        "dtype": np.dtype(dtype).name,
        "boundaries": boundaries,
        "next_chunk": 0,
        "token_count": 0,
        "processed_input_bytes": 0,
        "elapsed_seconds": 0.0,
        "complete": False,
    }


def _validate_resume(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    identity_fields = {
        "version",
        "input_path",
        "input_size",
        "input_mtime_ns",
        "output_path",
        "vocab_path",
        "vocab_sha256",
        "merges_path",
        "merges_sha256",
        "special_token",
        "dtype",
        "boundaries",
    }
    changed = {field for field in identity_fields if expected[field] != actual.get(field)}
    if changed:
        raise ValueError(f"cannot resume because inputs/settings changed: {', '.join(sorted(changed))}")


def encode_corpus(
    input_path: str | Path,
    output_path: str | Path,
    vocab_path: str | Path,
    merges_path: str | Path,
    special_token: str = "<|endoftext|>",
    dtype: str = "uint16",
    workers: int | None = None,
    chunks_per_worker: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Encode a corpus to a raw array, resuming only from committed chunk boundaries."""

    source = Path(input_path)
    destination = Path(output_path)
    vocab = Path(vocab_path)
    merges = Path(merges_path)
    output_dtype = np.dtype(dtype)
    tokenizer = Tokenizer.from_files(vocab, merges, [special_token])
    max_token_id = max(tokenizer.vocab)
    if max_token_id > np.iinfo(output_dtype).max:
        raise ValueError(f"token ID {max_token_id} does not fit in {output_dtype}")

    worker_count = workers or min(os.cpu_count() or 4, 8)
    desired_chunks = max(1, worker_count * chunks_per_worker)
    with source.open("rb") as handle:
        boundaries = find_chunk_boundaries(handle, desired_chunks, special_token.encode("utf-8"))
    manifest_path = destination.with_name(f"{destination.name}.progress.json")
    expected = _build_manifest(
        source,
        destination,
        vocab,
        merges,
        special_token,
        output_dtype.name,
        boundaries,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        manifest = expected
        with destination.open("wb"):
            pass
        _write_json_atomic(manifest_path, manifest)
    elif manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume(expected, manifest)
        if not destination.exists():
            raise FileNotFoundError(f"resume output is missing: {destination}")
        committed_bytes = int(manifest["token_count"]) * output_dtype.itemsize
        with destination.open("r+b") as handle:
            handle.truncate(committed_bytes)
    elif destination.exists():
        raise FileExistsError(
            f"{destination} exists without a progress manifest; pass --overwrite to replace it"
        )
    else:
        manifest = expected
        with destination.open("wb"):
            pass
        _write_json_atomic(manifest_path, manifest)

    if manifest["complete"]:
        return manifest

    ranges = list(zip(boundaries[:-1], boundaries[1:]))
    next_chunk = int(manifest["next_chunk"])
    tasks = [
        (index, str(source), start, end, output_dtype.name)
        for index, (start, end) in enumerate(ranges)
        if index >= next_chunk
    ]
    started = time.perf_counter()
    previous_elapsed = float(manifest["elapsed_seconds"])
    initializer_args = (str(vocab), str(merges), [special_token])

    if worker_count == 1:
        _initialize_worker(*initializer_args)
        results = map(_encode_chunk, tasks)
        pool = None
    else:
        pool = multiprocessing.get_context("spawn").Pool(
            processes=min(worker_count, len(tasks)),
            initializer=_initialize_worker,
            initargs=initializer_args,
        )
        results = pool.imap(_encode_chunk, tasks, chunksize=1)

    try:
        with destination.open("ab") as output:
            for index, token_ids, input_bytes in results:
                if index != manifest["next_chunk"]:
                    raise RuntimeError(f"out-of-order chunk {index}; expected {manifest['next_chunk']}")
                token_ids.tofile(output)
                output.flush()
                os.fsync(output.fileno())
                manifest["next_chunk"] = index + 1
                manifest["token_count"] += int(token_ids.size)
                manifest["processed_input_bytes"] += input_bytes
                manifest["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
                manifest["complete"] = manifest["next_chunk"] == len(ranges)
                _write_json_atomic(manifest_path, manifest)
                mib_done = manifest["processed_input_bytes"] / (1024 * 1024)
                rate = mib_done / max(manifest["elapsed_seconds"], 1e-9)
                print(
                    f"chunk {manifest['next_chunk']}/{len(ranges)}: "
                    f"{manifest['token_count']:,} tokens, {mib_done:,.1f} MiB, {rate:,.2f} MiB/s",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    manifest["bytes_per_token"] = manifest["input_size"] / manifest["token_count"]
    manifest["output_bytes"] = destination.stat().st_size
    _write_json_atomic(manifest_path, manifest)
    metadata_path = destination.with_name(f"{destination.name}.meta.json")
    _write_json_atomic(metadata_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--special-token", default="<|endoftext|>")
    parser.add_argument("--dtype", default="uint16")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--chunks-per-worker", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = encode_corpus(
        args.input,
        args.output,
        args.vocab,
        args.merges,
        special_token=args.special_token,
        dtype=args.dtype,
        workers=args.workers,
        chunks_per_worker=args.chunks_per_worker,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
