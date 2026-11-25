"""Minimal final-block CLIP adaptation controls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import ModelBatch
from .foundation import clip_feature_tensor
from .fusion import _ProjectedHead


def configure_final_block_adaptation(model: nn.Module) -> tuple[str, ...]:
    """Freeze CLIP, then unfreeze both final encoder blocks and projections."""

    model.requires_grad_(False)
    try:
        modules = (
            model.vision_model.encoder.layers[-1],
            model.text_model.encoder.layers[-1],
            model.visual_projection,
            model.text_projection,
        )
    except (AttributeError, IndexError) as error:
        raise ValueError("model does not expose the expected CLIP encoder structure") from error
    for module in modules:
        module.requires_grad_(True)
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def parameter_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def changed_parameter_names(model: nn.Module, before: Mapping[str, Tensor]) -> tuple[str, ...]:
    current = dict(model.named_parameters())
    if current.keys() != before.keys():
        raise ValueError("parameter names changed after the snapshot")
    return tuple(
        name for name, old in before.items() if not current[name].detach().cpu().equal(old)
    )


class AdaptiveCLIPClassifier(nn.Module):
    """Final-block CLIP adaptation on raw images/text; no final-embedding cache."""

    def __init__(self, model: nn.Module, processor: Any, fusion: _ProjectedHead) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.fusion = fusion
        self.trainable_encoder_names = configure_final_block_adaptation(model)

    def forward(self, batch: ModelBatch) -> Tensor:
        device = next(self.model.parameters()).device
        inputs = self.processor(
            images=batch.modalities["image"].detach().cpu(),
            text=list(batch.modalities["text"]),
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        embeddings = {
            "image": F.normalize(
                clip_feature_tensor(
                    self.model.get_image_features(pixel_values=inputs["pixel_values"])
                ),
                dim=1,
            ),
            "text": F.normalize(
                clip_feature_tensor(
                    self.model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )
                ),
                dim=1,
            ),
        }
        availability = {name: batch.availability[name] for name in self.fusion.modalities}
        return self.fusion(embeddings, availability)
