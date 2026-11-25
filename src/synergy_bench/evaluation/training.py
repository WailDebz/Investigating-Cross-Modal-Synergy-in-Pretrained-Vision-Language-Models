"""Small deterministic binary training loop with validation checkpointing."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainingSummary:
    epochs: int
    best_validation_loss: float
    training_seconds: float
    peak_device_bytes: int | None


def _trainable_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state(model: nn.Module, state: dict[str, Tensor]) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def _loss(model: nn.Module, batches: Sequence[Any], forward: Callable) -> Tensor:
    total: Tensor | None = None
    count = 0
    for batch in batches:
        logits = forward(model, batch)
        targets = batch.targets.to(device=logits.device, dtype=logits.dtype)
        if logits.shape != targets.shape:
            raise ValueError("binary logits and targets must share shape [batch]")
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
        total = loss if total is None else total + loss
        count += targets.numel()
    if total is None:
        raise ValueError("training and validation batches must be non-empty")
    return total / count


def fit_binary(
    model: nn.Module,
    train_batches: Sequence[Any],
    validation_batches: Sequence[Any],
    *,
    forward: Callable[[nn.Module, Any], Tensor] = lambda module, batch: module(batch),
    epochs: int = 25,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 0,
) -> TrainingSummary:
    if not train_batches or not validation_batches:
        raise ValueError("training and validation batches must be non-empty")
    if (
        epochs <= 0
        or learning_rate <= 0
        or patience <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("epochs, learning_rate, and patience must be positive")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    device = trainable[0].device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state = _trainable_state(model)
    best_loss = float("inf")
    stale_epochs = 0
    completed_epochs = 0
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_batches), generator=generator).tolist()
        for index in order:
            loss = _loss(model, [train_batches[index]], forward)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(_loss(model, validation_batches, forward))
        completed_epochs = epoch + 1
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = _trainable_state(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    _restore_trainable_state(model, best_state)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return TrainingSummary(completed_epochs, best_loss, time.perf_counter() - started, peak)


@torch.inference_mode()
def predict_logits(
    model: nn.Module,
    batches: Sequence[Any],
    forward: Callable[[nn.Module, Any], Tensor] = lambda module, batch: module(batch),
) -> tuple[Tensor, Tensor]:
    if not batches:
        raise ValueError("batches must be non-empty")
    model.eval()
    logits, targets = [], []
    for batch in batches:
        output = forward(model, batch)
        logits.append(output.detach().cpu())
        targets.append(batch.targets.detach().cpu())
    return torch.cat(logits), torch.cat(targets)
