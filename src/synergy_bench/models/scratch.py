"""Small from-scratch image and text encoders."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .contracts import ModalityEncoder, ModelBatch
from .fusion import _ProjectedHead


_TOKEN = re.compile(r"[a-z0-9]+|[^\w\s]")


class TextVocabulary:
    PAD = 0
    UNK = 1

    def __init__(self, tokens: Sequence[str], max_length: int = 32) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.token_to_id = {token: index + 2 for index, token in enumerate(tokens)}
        self.max_length = max_length

    @classmethod
    def from_texts(cls, texts: Sequence[str], max_length: int = 32) -> "TextVocabulary":
        counts = Counter(token for text in texts for token in _TOKEN.findall(text.lower()))
        return cls(sorted(counts), max_length=max_length)

    def encode_batch(self, texts: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        tokenized = [_TOKEN.findall(text.lower())[: self.max_length] for text in texts]
        length = max(1, max((len(tokens) for tokens in tokenized), default=0))
        ids = torch.full((len(texts), length), self.PAD, dtype=torch.long, device=device)
        mask = torch.zeros((len(texts), length), dtype=torch.bool, device=device)
        for row, tokens in enumerate(tokenized):
            if tokens:
                ids[row, : len(tokens)] = torch.tensor(
                    [self.token_to_id.get(token, self.UNK) for token in tokens], device=device
                )
                mask[row, : len(tokens)] = True
        return ids, mask

    def __len__(self) -> int:
        return len(self.token_to_id) + 2


class ScratchImageEncoder(ModalityEncoder):
    def __init__(self, output_dim: int = 64) -> None:
        super().__init__(output_dim)
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("image input must be [batch, 3, height, width]")
        return self.network(inputs)


class ScratchTextEncoder(ModalityEncoder):
    def __init__(self, vocabulary_size: int, output_dim: int = 64, embedding_dim: int = 32) -> None:
        super().__init__(output_dim)
        if vocabulary_size < 2 or embedding_dim <= 0:
            raise ValueError("invalid text encoder dimensions")
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.projection = nn.Sequential(nn.Linear(embedding_dim, output_dim), nn.ReLU())

    def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        token_ids, mask = inputs
        if token_ids.shape != mask.shape or token_ids.ndim != 2:
            raise ValueError("token ids and mask must share shape [batch, tokens]")
        embedded = self.embedding(token_ids)
        weights = mask.to(embedded.dtype)[:, :, None]
        pooled = (embedded * weights).sum(1) / weights.sum(1).clamp_min(1)
        return self.projection(pooled)


class ScratchClassifier(nn.Module):
    def __init__(
        self,
        fusion: _ProjectedHead,
        vocabulary: TextVocabulary | None = None,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.modalities = fusion.modalities
        self.fusion = fusion
        self.vocabulary = vocabulary
        if any(name not in {"image", "text"} for name in self.modalities):
            raise ValueError("scratch modalities must be image and/or text")
        if any(fusion.input_dims[name] != embedding_dim for name in self.modalities):
            raise ValueError("fusion input dimensions must match embedding_dim")
        self.encoders = nn.ModuleDict()
        if "image" in self.modalities:
            self.encoders["image"] = ScratchImageEncoder(embedding_dim)
        if "text" in self.modalities:
            if vocabulary is None:
                raise ValueError("text models require a training-set vocabulary")
            self.encoders["text"] = ScratchTextEncoder(len(vocabulary), embedding_dim)

    def forward(self, batch: ModelBatch) -> Tensor:
        embeddings: dict[str, Tensor] = {}
        if "image" in self.modalities:
            embeddings["image"] = self.encoders["image"](batch.modalities["image"])
        if "text" in self.modalities:
            device = next(self.encoders["text"].parameters()).device
            tokens = self.vocabulary.encode_batch(batch.modalities["text"], device)
            embeddings["text"] = self.encoders["text"](tokens)
        availability = {name: batch.availability[name] for name in self.modalities}
        return self.fusion(embeddings, availability)


def scratch_input_dims(modalities: Sequence[str], embedding_dim: int = 64) -> Mapping[str, int]:
    if not modalities or any(name not in {"image", "text"} for name in modalities):
        raise ValueError("scratch modalities must be image and/or text")
    return {name: embedding_dim for name in modalities}
