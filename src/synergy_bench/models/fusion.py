"""Shared image-text fusion heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .contracts import FusionHead
from .contracts import ModelBatch


class _ProjectedHead(FusionHead):
    def __init__(self, input_dims: Mapping[str, int]) -> None:
        super().__init__()
        if not input_dims or any(not name or dim <= 0 for name, dim in input_dims.items()):
            raise ValueError("input_dims must contain positive dimensions")
        self.modalities = tuple(input_dims)
        self.input_dims = dict(input_dims)

    def _inputs(
        self,
        embeddings: Mapping[str, Tensor],
        availability: Mapping[str, Tensor] | None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        values: list[Tensor] = []
        masks: list[Tensor] = []
        batch_size: int | None = None
        for modality in self.modalities:
            if modality not in embeddings:
                raise ValueError(f"missing embedding for {modality!r}")
            value = embeddings[modality]
            if value.ndim != 2 or value.shape[1] != self.input_dims[modality]:
                raise ValueError(
                    f"{modality!r} embedding must be [batch, {self.input_dims[modality]}]"
                )
            batch_size = value.shape[0] if batch_size is None else batch_size
            if value.shape[0] != batch_size:
                raise ValueError("all embeddings must have the same batch size")
            if availability is None:
                mask = torch.ones(value.shape[0], 1, dtype=value.dtype, device=value.device)
            else:
                if modality not in availability:
                    raise ValueError(f"missing availability mask for {modality!r}")
                mask = availability[modality].to(device=value.device, dtype=value.dtype)
                if mask.shape != (value.shape[0],):
                    raise ValueError(f"availability[{modality!r}] must be [batch]")
                mask = mask[:, None]
            values.append(value)
            masks.append(mask)
        if torch.cat(masks, dim=1).sum(1).eq(0).any():
            raise ValueError("every sample must have at least one available modality")
        return values, masks


class LinearFusionHead(_ProjectedHead):
    def __init__(self, input_dims: Mapping[str, int]) -> None:
        super().__init__(input_dims)
        self.classifier = nn.Linear(sum(input_dims.values()), 1)

    def forward(self, embeddings, availability=None):
        values, masks = self._inputs(embeddings, availability)
        return self.classifier(torch.cat([value * mask for value, mask in zip(values, masks)], 1))[
            :, 0
        ]


class MLPFusionHead(_ProjectedHead):
    def __init__(self, input_dims: Mapping[str, int], hidden_dim: int = 64) -> None:
        super().__init__(input_dims)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.classifier = nn.Sequential(
            nn.Linear(sum(input_dims.values()), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings, availability=None):
        values, masks = self._inputs(embeddings, availability)
        return self.classifier(torch.cat([value * mask for value, mask in zip(values, masks)], 1))[
            :, 0
        ]


class BilinearFusionHead(_ProjectedHead):
    def __init__(self, input_dims: Mapping[str, int], projection_dim: int = 16) -> None:
        super().__init__(input_dims)
        if len(input_dims) != 2:
            raise ValueError("exact bilinear fusion is restricted to two modalities")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        self.projection_dim = projection_dim
        self.tensor_dim = (projection_dim + 1) ** 2
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, projection_dim) for name, dim in input_dims.items()}
        )
        self.classifier = nn.Linear(self.tensor_dim, 1)

    def forward(self, embeddings, availability=None):
        values, masks = self._inputs(embeddings, availability)
        augmented = []
        for modality, value, mask in zip(self.modalities, values, masks):
            projected = torch.tanh(self.projections[modality](value)) * mask
            augmented.append(torch.cat([torch.ones_like(mask), projected], 1))
        tensor = (augmented[0][:, :, None] * augmented[1][:, None, :]).flatten(1)
        return self.classifier(tensor)[:, 0]


class LateFusionHead(_ProjectedHead):
    """Trainable per-modality calibration followed by available-logit averaging."""

    def __init__(self, modalities: Sequence[str]) -> None:
        if not modalities or len(set(modalities)) != len(modalities):
            raise ValueError("modalities must be non-empty and unique")
        super().__init__({name: 1 for name in modalities})
        self.log_scales = nn.Parameter(torch.zeros(len(modalities)))
        self.biases = nn.Parameter(torch.zeros(len(modalities)))

    def forward(self, embeddings, availability=None):
        values, masks = self._inputs(embeddings, availability)
        calibrated = []
        for index, (value, mask) in enumerate(zip(values, masks)):
            logit = value[:, 0] * self.log_scales[index].exp() + self.biases[index]
            calibrated.append(logit * mask[:, 0])
        denominator = torch.cat(masks, 1).sum(1)
        return torch.stack(calibrated, 1).sum(1) / denominator


class EmbeddingClassifier(nn.Module):
    """Apply a fusion head to already-extracted modality embeddings."""

    def __init__(self, fusion: _ProjectedHead) -> None:
        super().__init__()
        self.fusion = fusion

    def forward(self, batch: ModelBatch) -> Tensor:
        embeddings = {name: batch.modalities[name] for name in self.fusion.modalities}
        availability = {name: batch.availability[name] for name in self.fusion.modalities}
        return self.fusion(embeddings, availability)
