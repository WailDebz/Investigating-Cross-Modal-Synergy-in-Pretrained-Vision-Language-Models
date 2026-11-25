"""Frozen CLIP image/text feature extraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import ModelBatch


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def clip_feature_tensor(output: Any) -> Tensor:
    """Support tensor-returning Transformers 4 and pooled-output Transformers 5."""

    if isinstance(output, Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, Tensor):
        return pooled
    raise TypeError("CLIP feature method returned an unsupported value")


class FrozenCLIP(nn.Module):
    def __init__(self, model: nn.Module, processor: Any, model_id: str, revision: str) -> None:
        super().__init__()
        if not model_id or not revision or revision == "main" or revision.startswith("REPLACE_"):
            raise ValueError("CLIP model_id and immutable revision are required")
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.revision = revision
        self.model.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        revision: str,
        model_id: str = DEFAULT_CLIP_MODEL,
        local_files_only: bool = False,
    ) -> "FrozenCLIP":
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError("transformers is required for CLIP") from error
        model = CLIPModel.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
        processor = CLIPProcessor.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
        return cls(model, processor, model_id, revision)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def encode_batch(self, batch: ModelBatch, normalize: bool = True) -> dict[str, Tensor]:
        inputs = self.processor(
            images=batch.modalities["image"].detach().cpu(),
            text=list(batch.modalities["text"]),
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        image = clip_feature_tensor(
            self.model.get_image_features(pixel_values=inputs["pixel_values"])
        )
        text = clip_feature_tensor(
            self.model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            )
        )
        if normalize:
            image, text = F.normalize(image, dim=1), F.normalize(text, dim=1)
        return {"image": image, "text": text}

    @torch.inference_mode()
    def zero_shot_image_logits(
        self, images: Tensor, class_prompts: Sequence[str]
    ) -> Tensor:
        if len(class_prompts) < 2:
            raise ValueError("zero-shot classification requires at least two class prompts")
        inputs = self.processor(
            images=images.detach().cpu(),
            text=list(class_prompts),
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        image = F.normalize(
            clip_feature_tensor(
                self.model.get_image_features(pixel_values=inputs["pixel_values"])
            ),
            dim=1,
        )
        text = F.normalize(
            clip_feature_tensor(
                self.model.get_text_features(
                    input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
                )
            ),
            dim=1,
        )
        return self.model.logit_scale.exp() * image @ text.T
