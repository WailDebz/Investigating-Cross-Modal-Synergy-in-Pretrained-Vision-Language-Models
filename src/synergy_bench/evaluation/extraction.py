"""One-pass frozen embedding extraction and cache reuse."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from synergy_bench.models.contracts import ModelBatch
from synergy_bench.models.foundation import FrozenCLIP

from .cache import CacheMetadata, EmbeddingCache


def extract_to_cache(
    encoder: FrozenCLIP,
    batches: Sequence[ModelBatch],
    cache: EmbeddingCache,
    metadata: Mapping[str, CacheMetadata],
) -> tuple[tuple[str, ...], dict[str, Tensor], bool]:
    if not batches or set(metadata) != {"image", "text"}:
        raise ValueError("image and text metadata plus non-empty batches are required")
    if any(metadata[name].modality != name for name in metadata):
        raise ValueError("cache metadata modality mismatch")
    sample_ids = tuple(sample_id for batch in batches for sample_id in batch.sample_ids)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be unique across extraction batches")
    try:
        cached = {name: cache.load(values, sample_ids) for name, values in metadata.items()}
        return sample_ids, cached, True
    except FileNotFoundError:
        extracted: dict[str, list[Tensor]] = {"image": [], "text": []}
        for batch in batches:
            features = encoder.encode_batch(batch)
            for name in extracted:
                extracted[name].append(features[name].detach().cpu())
        combined = {name: torch.cat(parts) for name, parts in extracted.items()}
        for name, values in metadata.items():
            cache.save(values, sample_ids, combined[name])
        return sample_ids, combined, False
