"""Loss, batching, and serialization helpers for Assignment 1 §4–5."""

from __future__ import annotations

import os
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean next-token negative log-likelihood from unnormalized logits."""

    max_logits = logits.amax(dim=-1, keepdim=True)
    shifted = logits - max_logits
    log_normalizer = shifted.exp().sum(dim=-1).log() + max_logits.squeeze(-1)
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample random contiguous inputs and their one-token-shifted targets."""

    num_starts = len(dataset) - context_length
    if num_starts <= 0:
        raise ValueError("dataset must contain more than context_length tokens")
    starts = torch.randint(0, num_starts, (batch_size,), generator=generator)
    input_array = np.stack([dataset[start : start + context_length] for start in starts.tolist()])
    target_array = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts.tolist()])
    inputs = torch.from_numpy(input_array).to(device=device, dtype=torch.long)
    targets = torch.from_numpy(target_array).to(device=device, dtype=torch.long)
    return inputs, targets


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize all state required to resume an optimization run."""

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model, optimizer moments, and iteration; return the resume step."""

    first_parameter = next(model.parameters(), None)
    map_location = None if first_parameter is None else first_parameter.device
    checkpoint = torch.load(src, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
