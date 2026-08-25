"""Configurable A1 language-model training, evaluation, logging, and resume CLI.

The official handout requires memory-mapped data, periodic validation, and resumable
checkpoints (§5.3). This module keeps those orchestration concerns separate from the
model, optimizer, and thin test adapters.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import time
import tomllib
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cs336_basics.nn import TransformerLM
from cs336_basics.optim import AdamW, clip_gradients, cosine_learning_rate
from cs336_basics.training import cross_entropy, get_batch


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1_344
    rope_theta: float = 10_000.0
    normalization: str = "pre"
    positional_encoding: str = "rope"
    feed_forward: str = "swiglu"
    tie_embeddings: bool = False
    tied_embedding_std: float = 0.02
    embedding_scale: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    loss_chunk_size: int | None = None
    eval_batch_size: int | None = None
    precision: str = "float32"
    max_steps: int = 5_000
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    cosine_cycle_steps: int = 5_000
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 500
    seed: int = 42


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto` to CUDA, then MPS, then CPU."""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, precision: str):
    """Autocast matrix-heavy forward work while retaining FP32 parameters/state."""

    if precision == "float32":
        return nullcontext()
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if precision not in dtypes:
        raise ValueError("precision must be 'float32', 'bfloat16', or 'float16'")
    return torch.autocast(device_type=device.type, dtype=dtypes[precision])


def load_token_array(path: str | Path, dtype: str = "uint16") -> np.ndarray:
    """Open a 1D token array lazily from `.npy` or a raw binary file."""

    token_path = Path(path)
    if token_path.suffix == ".npy":
        tokens = np.load(token_path, mmap_mode="r")
    else:
        tokens = np.memmap(token_path, mode="r", dtype=np.dtype(dtype))
    if tokens.ndim != 1:
        raise ValueError(f"expected a 1D token array, got shape {tokens.shape}")
    return tokens


def build_model(config: ModelConfig, device: torch.device) -> TransformerLM:
    return TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=device,
        normalization=config.normalization,
        positional_encoding=config.positional_encoding,
        feed_forward=config.feed_forward,
        tie_embeddings=config.tie_embeddings,
        tied_embedding_std=config.tied_embedding_std,
        embedding_scale=config.embedding_scale,
    )


def train_step(
    model: TransformerLM,
    optimizer: AdamW,
    inputs: Tensor,
    targets: Tensor,
    learning_rate: float,
    max_grad_norm: float,
) -> tuple[float, float]:
    """Run one ordered zero-grad/forward/backward/clip/update step."""

    return train_accumulated_step(
        model,
        optimizer,
        [(inputs, targets)],
        learning_rate,
        max_grad_norm,
        loss_chunk_size=None,
    )


def _backward_language_model_loss(
    model: TransformerLM,
    inputs: Tensor,
    targets: Tensor,
    *,
    loss_scale: float,
    loss_chunk_size: int | None,
    precision: str,
) -> float:
    """Backpropagate one microbatch, optionally materializing logits in token chunks."""

    if loss_chunk_size is None:
        with autocast_context(inputs.device, precision):
            loss = cross_entropy(model(inputs), targets)
        (loss * loss_scale).backward()
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite training loss: {loss_value}")
        return loss_value
    if loss_chunk_size <= 0:
        raise ValueError("loss_chunk_size must be positive")

    # Detach at the final hidden states so each vocabulary-sized chunk can release
    # its logits immediately. Accumulate dL/dhidden, then traverse the Transformer
    # graph exactly once after all LM-head chunks have backpropagated.
    with autocast_context(inputs.device, precision):
        hidden = model.forward_hidden(inputs)
    hidden_leaf = hidden.detach().requires_grad_(True)
    flat_hidden = hidden_leaf.flatten(0, -2)
    flat_targets = targets.flatten()
    token_count = flat_targets.numel()
    total_loss = torch.zeros((), device=hidden.device)
    for start in range(0, token_count, loss_chunk_size):
        end = min(start + loss_chunk_size, token_count)
        with autocast_context(inputs.device, precision):
            chunk_loss = cross_entropy(model.lm_head(flat_hidden[start:end]), flat_targets[start:end])
        chunk_weight = (end - start) / token_count
        (chunk_loss * chunk_weight * loss_scale).backward()
        # Keep metric reduction on-device. Pulling every chunk to CPU would force
        # dozens of MPS synchronizations per optimizer update.
        total_loss.add_(chunk_loss.detach(), alpha=chunk_weight)
    if hidden_leaf.grad is None:
        raise RuntimeError("chunked loss did not produce hidden-state gradients")
    hidden.backward(hidden_leaf.grad)
    loss_value = float(total_loss.cpu())
    if not math.isfinite(loss_value):
        raise RuntimeError(f"non-finite training loss: {loss_value}")
    return loss_value


def train_accumulated_step(
    model: TransformerLM,
    optimizer: AdamW,
    batches: list[tuple[Tensor, Tensor]],
    learning_rate: float,
    max_grad_norm: float,
    *,
    loss_chunk_size: int | None,
    precision: str = "float32",
) -> tuple[float, float]:
    """Average microbatch gradients, then clip and perform one AdamW update."""

    if not batches:
        raise ValueError("at least one microbatch is required")
    model.train()
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    loss_scale = 1.0 / len(batches)
    losses = [
        _backward_language_model_loss(
            model,
            inputs,
            targets,
            loss_scale=loss_scale,
            loss_chunk_size=loss_chunk_size,
            precision=precision,
        )
        for inputs, targets in batches
    ]
    # Clipping already computes the global norm. Reusing it avoids a second set
    # of per-parameter reductions and CPU synchronizations solely for logging.
    grad_norm = float(clip_gradients(model.parameters(), max_grad_norm).cpu())
    optimizer.step()
    return sum(losses) / len(losses), grad_norm


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    num_batches: int,
    seed: int,
    precision: str = "float32",
) -> float:
    """Average validation loss on the same deterministic sample set each time."""

    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for _ in range(num_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device, generator)
        with autocast_context(device, precision):
            loss = cross_entropy(model(inputs), targets)
        losses.append(float(loss.cpu()))
    model.train(was_training)
    return sum(losses) / len(losses)


def _checkpoint_payload(
    model: TransformerLM,
    optimizer: AdamW,
    iteration: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    train_generator: torch.Generator,
    best_validation_loss: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "train_generator_state": train_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "best_validation_loss": best_validation_loss,
        "elapsed_seconds": elapsed_seconds,
    }


def save_run_checkpoint(
    path: str | Path,
    model: TransformerLM,
    optimizer: AdamW,
    iteration: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    train_generator: torch.Generator,
    best_validation_loss: float,
    elapsed_seconds: float = 0.0,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            iteration,
            model_config,
            train_config,
            train_generator,
            best_validation_loss,
            elapsed_seconds,
        ),
        checkpoint_path,
    )


def load_run_checkpoint(
    path: str | Path,
    model: TransformerLM,
    optimizer: AdamW,
    model_config: ModelConfig,
    train_config: TrainConfig,
    train_generator: torch.Generator,
    allow_precision_change: bool = False,
) -> tuple[int, float, float]:
    checkpoint = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    saved_model_config = asdict(ModelConfig(**checkpoint["model_config"]))
    if saved_model_config != asdict(model_config):
        raise ValueError("checkpoint model configuration does not match the requested model")
    trajectory_fields = {
        "batch_size",
        "gradient_accumulation_steps",
        "loss_chunk_size",
        "precision",
        "max_learning_rate",
        "min_learning_rate",
        "warmup_steps",
        "cosine_cycle_steps",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "max_grad_norm",
        "seed",
    }
    requested_train_config = asdict(train_config)
    saved_train_config = asdict(TrainConfig(**checkpoint["train_config"]))
    changed_fields = {
        field
        for field in trajectory_fields
        if saved_train_config[field] != requested_train_config[field]
    }
    if allow_precision_change:
        changed_fields.discard("precision")
    if changed_fields:
        changed = ", ".join(sorted(changed_fields))
        raise ValueError(f"resume would change trajectory-defining settings: {changed}")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # ``map_location`` moves every tensor in the payload, including RNG state.
    # CPU generators reject MPS/CUDA byte tensors even though model and optimizer
    # tensors should load on the training device.
    train_generator.set_state(checkpoint["train_generator_state"].cpu())
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    return (
        int(checkpoint["iteration"]),
        float(checkpoint["best_validation_loss"]),
        float(checkpoint.get("elapsed_seconds", 0.0)),
    )


def _record_metric(path: Path, metric: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
    print(json.dumps(metric, sort_keys=True), flush=True)


def train(
    train_data: np.ndarray,
    validation_data: np.ndarray,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    output_dir: str | Path,
    resume: str | Path | None = None,
    allow_precision_change: bool = False,
) -> Path:
    """Train through `max_steps`; return the final checkpoint path."""

    if train_config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if train_config.loss_chunk_size is not None and train_config.loss_chunk_size <= 0:
        raise ValueError("loss_chunk_size must be positive")
    if train_config.precision not in {"float32", "bfloat16", "float16"}:
        raise ValueError("precision must be 'float32', 'bfloat16', or 'float16'")
    torch.manual_seed(train_config.seed)
    eval_batch_size = train_config.eval_batch_size or train_config.batch_size
    model = build_model(model_config, device)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.max_learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        eps=train_config.epsilon,
        weight_decay=train_config.weight_decay,
    )
    train_generator = torch.Generator(device="cpu").manual_seed(train_config.seed)
    iteration = 0
    best_validation_loss = math.inf
    elapsed_before_resume = 0.0
    if resume is not None:
        iteration, best_validation_loss, elapsed_before_resume = load_run_checkpoint(
            resume,
            model,
            optimizer,
            model_config,
            train_config,
            train_generator,
            allow_precision_change,
        )

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    _record_metric(
        metrics_path,
        {
            "event": "start" if resume is None else "resume",
            "iteration": iteration,
            "elapsed_before_resume": elapsed_before_resume,
            "device": str(device),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
        },
    )

    if iteration == 0:
        initial_validation_loss = evaluate(
            model,
            validation_data,
            eval_batch_size,
            model_config.context_length,
            device,
            train_config.eval_batches,
            train_config.seed + 1,
            train_config.precision,
        )
        best_validation_loss = initial_validation_loss
        _record_metric(
            metrics_path,
            {"event": "validation", "iteration": 0, "validation_loss": initial_validation_loss},
        )

    training_started = time.perf_counter()
    interval_started = training_started
    while iteration < train_config.max_steps:
        step_started = time.perf_counter()
        learning_rate = cosine_learning_rate(
            iteration,
            train_config.max_learning_rate,
            train_config.min_learning_rate,
            train_config.warmup_steps,
            train_config.cosine_cycle_steps,
        )
        batches = [
            get_batch(
                train_data,
                train_config.batch_size,
                model_config.context_length,
                device,
                train_generator,
            )
            for _ in range(train_config.gradient_accumulation_steps)
        ]
        training_loss, grad_norm = train_accumulated_step(
            model,
            optimizer,
            batches,
            learning_rate,
            train_config.max_grad_norm,
            loss_chunk_size=train_config.loss_chunk_size,
            precision=train_config.precision,
        )
        iteration += 1
        step_seconds = time.perf_counter() - step_started

        if iteration % train_config.log_interval == 0 or iteration == 1:
            tokens = (
                train_config.batch_size
                * train_config.gradient_accumulation_steps
                * model_config.context_length
            )
            _record_metric(
                metrics_path,
                {
                    "event": "train",
                    "iteration": iteration,
                    "training_loss": training_loss,
                    "learning_rate": learning_rate,
                    "gradient_norm": grad_norm,
                    "step_seconds": step_seconds,
                    "tokens_per_second": tokens / step_seconds,
                    "wall_seconds": elapsed_before_resume + time.perf_counter() - training_started,
                    "interval_seconds": time.perf_counter() - interval_started,
                },
            )
            interval_started = time.perf_counter()

        should_evaluate = iteration % train_config.eval_interval == 0 or iteration == train_config.max_steps
        if should_evaluate:
            validation_loss = evaluate(
                model,
                validation_data,
                eval_batch_size,
                model_config.context_length,
                device,
                train_config.eval_batches,
                train_config.seed + 1,
                train_config.precision,
            )
            _record_metric(
                metrics_path,
                {
                    "event": "validation",
                    "iteration": iteration,
                    "validation_loss": validation_loss,
                    "wall_seconds": elapsed_before_resume + time.perf_counter() - training_started,
                },
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_run_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer,
                    iteration,
                    model_config,
                    train_config,
                    train_generator,
                    best_validation_loss,
                    elapsed_before_resume + time.perf_counter() - training_started,
                )

        if iteration % train_config.checkpoint_interval == 0:
            save_run_checkpoint(
                run_dir / f"step_{iteration:07d}.pt",
                model,
                optimizer,
                iteration,
                model_config,
                train_config,
                train_generator,
                best_validation_loss,
                elapsed_before_resume + time.perf_counter() - training_started,
            )

    final_path = run_dir / "last.pt"
    save_run_checkpoint(
        final_path,
        model,
        optimizer,
        iteration,
        model_config,
        train_config,
        train_generator,
        best_validation_loss,
        elapsed_before_resume + time.perf_counter() - training_started,
    )
    return final_path


def run_smoke_checks(device: torch.device = torch.device("cpu")) -> dict[str, float | bool]:
    """Prove single-batch memorization and exact optimizer-state resume on a tiny model."""

    torch.manual_seed(7)
    model_config = ModelConfig(
        vocab_size=32,
        context_length=12,
        d_model=32,
        num_layers=1,
        num_heads=4,
        d_ff=64,
        rope_theta=10_000.0,
    )
    model = build_model(model_config, device)
    sequence = torch.arange(13, device=device).repeat(4, 1)
    inputs, targets = sequence[:, :-1], sequence[:, 1:]
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0, betas=(0.9, 0.95))
    with torch.no_grad():
        initial_loss = float(cross_entropy(model(inputs), targets).cpu())
    for _ in range(150):
        train_step(model, optimizer, inputs, targets, learning_rate=1e-2, max_grad_norm=1.0)
    with torch.no_grad():
        final_loss = float(cross_entropy(model(inputs), targets).cpu())

    base = build_model(model_config, device)
    continuous = copy.deepcopy(base)
    resumed_prefix = copy.deepcopy(base)
    continuous_optimizer = AdamW(continuous.parameters(), lr=1e-3, weight_decay=0.01)
    prefix_optimizer = AdamW(resumed_prefix.parameters(), lr=1e-3, weight_decay=0.01)
    for _ in range(8):
        train_step(continuous, continuous_optimizer, inputs, targets, 1e-3, 1.0)
    for _ in range(4):
        train_step(resumed_prefix, prefix_optimizer, inputs, targets, 1e-3, 1.0)

    smoke_train_config = TrainConfig(max_steps=8)
    train_generator = torch.Generator(device="cpu").manual_seed(9)
    with tempfile.TemporaryDirectory() as temporary_dir:
        checkpoint_path = Path(temporary_dir) / "resume.pt"
        save_run_checkpoint(
            checkpoint_path,
            resumed_prefix,
            prefix_optimizer,
            4,
            model_config,
            smoke_train_config,
            train_generator,
            math.inf,
            0.0,
        )
        resumed = build_model(model_config, device)
        resumed_optimizer = AdamW(resumed.parameters(), lr=1e-3, weight_decay=0.01)
        resumed_generator = torch.Generator(device="cpu")
        loaded_iteration, _, _ = load_run_checkpoint(
            checkpoint_path,
            resumed,
            resumed_optimizer,
            model_config,
            smoke_train_config,
            resumed_generator,
        )
        for _ in range(loaded_iteration, 8):
            train_step(resumed, resumed_optimizer, inputs, targets, 1e-3, 1.0)

    max_resume_difference = max(
        float((left - right).detach().abs().max().cpu())
        for left, right in zip(continuous.parameters(), resumed.parameters())
    )
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "single_batch_overfit": final_loss < 0.05,
        "max_resume_parameter_difference": max_resume_difference,
        "resume_equivalent": max_resume_difference == 0.0,
    }


def load_profile_defaults(path: str | Path, profile_name: str) -> dict[str, Any]:
    """Load a flat CLI-default profile with optional single-parent inheritance."""

    profile_path = Path(path).resolve()
    with profile_path.open("rb") as handle:
        document = tomllib.load(handle)
    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"{profile_path} must contain a [profiles] table")

    def resolve(name: str, stack: tuple[str, ...] = ()) -> dict[str, Any]:
        if name in stack:
            chain = " -> ".join((*stack, name))
            raise ValueError(f"cyclic profile inheritance: {chain}")
        values = profiles.get(name)
        if not isinstance(values, dict):
            raise ValueError(f"unknown training profile: {name}")
        parent = values.get("extends")
        merged = resolve(parent, (*stack, name)) if isinstance(parent, str) else {}
        merged.update({key: value for key, value in values.items() if key != "extends"})
        return merged

    defaults = resolve(profile_name)
    # Profile paths are stable regardless of the caller's working directory.
    for key in ("train_data", "validation_data", "output_dir", "resume"):
        value = defaults.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            defaults[key] = str((profile_path.parent / value).resolve())
    return defaults


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="TOML file containing reusable [profiles.<name>] defaults")
    parser.add_argument("--profile", help="profile name from --config; explicit CLI flags override it")
    parser.add_argument("--smoke", action="store_true", help="run fast overfit/resume checks")
    parser.add_argument("--train-data")
    parser.add_argument("--validation-data")
    parser.add_argument("--data-dtype", default="uint16")
    parser.add_argument("--output-dir", default="runs/a1")
    parser.add_argument("--resume")
    parser.add_argument(
        "--allow-precision-change",
        action="store_true",
        help="explicitly permit resuming a checkpoint with a different autocast precision",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1_344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument(
        "--normalization",
        choices=("pre", "post", "none"),
        default="pre",
        help="select the baseline pre-norm, post-norm, or no-norm §7.3 architecture",
    )
    parser.add_argument(
        "--positional-encoding",
        choices=("rope", "none"),
        default="rope",
        help="use RoPE or the §7.3 NoPE ablation",
    )
    parser.add_argument(
        "--feed-forward",
        choices=("swiglu", "silu"),
        default="swiglu",
        help="use gated SwiGLU or the §7.3 ungated SiLU FFN ablation",
    )
    parser.add_argument(
        "--tie-embeddings",
        action="store_true",
        help="share input embedding and LM-head weights for the §7.5 modification",
    )
    parser.add_argument("--tied-embedding-std", type=float, default=0.02)
    parser.add_argument(
        "--embedding-scale",
        type=float,
        default=1.0,
        help="fixed input-only multiplier, useful when tied weights need a small output scale",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--loss-chunk-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--precision", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--cosine-cycle-steps", type=int)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = _parser()
    preliminary, _ = parser.parse_known_args()
    if preliminary.profile and not preliminary.config:
        parser.error("--profile requires --config")
    if preliminary.config:
        if not preliminary.profile:
            parser.error("--config requires --profile")
        try:
            profile_defaults = load_profile_defaults(preliminary.config, preliminary.profile)
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            parser.error(str(error))
        valid_destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(profile_defaults) - valid_destinations)
        if unknown:
            parser.error(f"unknown keys in profile {preliminary.profile}: {', '.join(unknown)}")
        parser.set_defaults(**profile_defaults)
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.smoke:
        print(json.dumps(run_smoke_checks(device), sort_keys=True))
        return
    if args.train_data is None or args.validation_data is None:
        parser.error("--train-data and --validation-data are required unless --smoke is used")

    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        normalization=args.normalization,
        positional_encoding=args.positional_encoding,
        feed_forward=args.feed_forward,
        tie_embeddings=args.tie_embeddings,
        tied_embedding_std=args.tied_embedding_std,
        embedding_scale=args.embedding_scale,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_chunk_size=args.loss_chunk_size,
        eval_batch_size=args.eval_batch_size,
        precision=args.precision,
        max_steps=args.max_steps,
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        cosine_cycle_steps=args.cosine_cycle_steps or args.max_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
    )
    train_tokens = load_token_array(args.train_data, args.data_dtype)
    validation_tokens = load_token_array(args.validation_data, args.data_dtype)
    final_checkpoint = train(
        train_tokens,
        validation_tokens,
        model_config,
        train_config,
        device,
        args.output_dir,
        args.resume,
        args.allow_precision_change,
    )
    print(f"final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
