"""Shared model-side contracts for Phase 1.2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import Tensor, nn


class ModalityEncoder(nn.Module, ABC):
    """A modality-specific module that returns one embedding per sample."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, inputs: Any) -> Tensor:
        """Return embeddings shaped ``[batch, output_dim]``."""


class FusionHead(nn.Module, ABC):
    """Combine named modality embeddings into model outputs."""

    @abstractmethod
    def forward(
        self,
        embeddings: Mapping[str, Tensor],
        availability: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Return one output row per input sample."""


@dataclass(frozen=True)
class ModelBatch:
    """Only the public fields that models may consume."""

    sample_ids: tuple[str, ...]
    modalities: Mapping[str, Any]
    targets: Tensor
    availability: Mapping[str, Tensor]


def adapt_batch(batch: Mapping[str, Any], device: str | torch.device = "cpu") -> ModelBatch:
    """Validate a collated Phase 1.1 batch and remove analysis metadata."""

    required = {"sample_id", "image", "text", "target", "availability"}
    missing = required - batch.keys()
    if missing:
        raise ValueError(f"batch is missing public fields: {sorted(missing)}")

    sample_ids = batch["sample_id"]
    texts = batch["text"]
    image = batch["image"]
    targets = batch["target"]
    availability = batch["availability"]

    if isinstance(sample_ids, str) or not isinstance(sample_ids, Sequence):
        raise TypeError("sample_id must be a sequence of strings")
    sample_ids = tuple(sample_ids)
    if not sample_ids or not all(isinstance(value, str) and value for value in sample_ids):
        raise ValueError("sample_id values must be non-empty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique within a batch")
    batch_size = len(sample_ids)

    if not isinstance(image, Tensor) or image.ndim != 4 or image.shape[0] != batch_size:
        raise ValueError("image must be a tensor shaped [batch, channels, height, width]")
    if isinstance(texts, str) or not isinstance(texts, Sequence) or len(texts) != batch_size:
        raise ValueError("text must contain one string per sample")
    if not all(isinstance(value, str) for value in texts):
        raise TypeError("text values must be strings")
    if not isinstance(targets, Tensor) or targets.shape != (batch_size,):
        raise ValueError("target must be a tensor shaped [batch]")
    if not isinstance(availability, Mapping):
        raise TypeError("availability must map modality names to masks")

    masks: dict[str, Tensor] = {}
    for modality in ("image", "text"):
        if modality not in availability:
            raise ValueError(f"availability is missing {modality!r}")
        mask = torch.as_tensor(availability[modality], dtype=torch.bool)
        if mask.shape != (batch_size,):
            raise ValueError(f"availability[{modality!r}] must be shaped [batch]")
        masks[modality] = mask.to(device)

    return ModelBatch(
        sample_ids=sample_ids,
        modalities={"image": image.to(device), "text": tuple(texts)},
        targets=targets.to(device),
        availability=masks,
    )


@dataclass(frozen=True)
class ParameterSummary:
    total: int
    trainable: int
    trainable_names: tuple[str, ...]


def summarize_parameters(module: nn.Module) -> ParameterSummary:
    """Report deduplicated total and trainable parameter counts."""

    parameters = tuple(module.named_parameters())
    return ParameterSummary(
        total=sum(parameter.numel() for _, parameter in parameters),
        trainable=sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad),
        trainable_names=tuple(name for name, parameter in parameters if parameter.requires_grad),
    )


@contextmanager
def seeded_model_init(seed: int) -> Iterator[None]:
    """Seed model construction without changing the caller's RNG state."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be non-negative")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


def _module_device(module: nn.Module) -> torch.device:
    devices = {value.device for value in (*module.parameters(), *module.buffers())}
    if len(devices) > 1:
        raise ValueError("inference requires all module parameters and buffers on one device")
    return next(iter(devices), torch.device("cpu"))


def _move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    return value


def inference(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    """Run inference with tensor inputs moved to the module's device."""

    device = _module_device(module)
    was_training = module.training
    module.eval()
    try:
        with torch.inference_mode():
            return module(
                *(_move_tensors(value, device) for value in args),
                **{key: _move_tensors(value, device) for key, value in kwargs.items()},
            )
    finally:
        module.train(was_training)
