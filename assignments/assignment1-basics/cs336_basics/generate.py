"""Autoregressive text generation from an A1 Transformer checkpoint.

The model produces next-token logits; decoding policy is deliberately kept here so
greedy, temperature, and nucleus sampling can be compared without changing weights.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from cs336_basics.bpe import Tokenizer
from cs336_basics.train import ModelConfig, build_model, resolve_device


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    stopped_at_eot: bool


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
    greedy: bool,
    generator: torch.Generator,
) -> int:
    """Select one token from a 1D logit vector using the requested policy."""

    if logits.ndim != 1:
        raise ValueError(f"expected 1D logits, got shape {tuple(logits.shape)}")
    if greedy:
        return int(torch.argmax(logits).item())
    if temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_ids = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        # Keep the first token that crosses top_p; remove only tokens whose
        # predecessors already reach the threshold.
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sorted_probabilities /= sorted_probabilities.sum()
        sampled_rank = torch.multinomial(
            sorted_probabilities.cpu(), 1, generator=generator
        )
        return int(sorted_ids.cpu()[sampled_rank].item())

    sampled = torch.multinomial(probabilities.cpu(), 1, generator=generator)
    return int(sampled.item())


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_p: float = 0.9,
    greedy: bool = False,
    seed: int = 42,
    stop_at_eot: bool = True,
    eot_token: str = "<|endoftext|>",
) -> GenerationResult:
    """Autoregressively append tokens, cropping model input to its context window."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("prompt must encode to at least one token")

    context_length = model.context_length
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    eot_bytes = eot_token.encode("utf-8")
    eot_id = next((token_id for token_id, piece in tokenizer.vocab.items() if piece == eot_bytes), None)
    generated = 0
    stopped_at_eot = False
    model.eval()

    for _ in range(max_new_tokens):
        context = token_ids[-context_length:]
        inputs = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(inputs)[0, -1]
        next_id = sample_next_token(
            logits,
            temperature=temperature,
            top_p=top_p,
            greedy=greedy,
            generator=generator,
        )
        if stop_at_eot and eot_id is not None and next_id == eot_id:
            stopped_at_eot = True
            break
        token_ids.append(next_id)
        generated += 1

    return GenerationResult(
        text=tokenizer.decode(token_ids),
        prompt_tokens=len(tokenizer.encode(prompt)),
        generated_tokens=generated,
        stopped_at_eot=stopped_at_eot,
    )


def load_model_from_checkpoint(path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load only the model/config portion of a full training checkpoint."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = build_model(model_config, torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    return model.to(device)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stop-at-eot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    model = load_model_from_checkpoint(args.checkpoint, device)
    result = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        greedy=args.greedy,
        seed=args.seed,
        stop_at_eot=args.stop_at_eot,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.text, encoding="utf-8")
    print(result.text)
    print(
        f"\n[generated_tokens={result.generated_tokens}, "
        f"stopped_at_eot={result.stopped_at_eot}, device={device}]"
    )


if __name__ == "__main__":
    main()
