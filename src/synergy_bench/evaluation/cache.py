"""Sample-aligned frozen embedding cache."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class CacheMetadata:
    sample_manifest_hash: str
    checkpoint: str
    revision: str
    preprocessing_hash: str
    code_hash: str
    config_hash: str
    dtype: str
    modality: str

    def __post_init__(self) -> None:
        if any(not value for value in asdict(self).values()):
            raise ValueError("all cache metadata fields are required")

    @property
    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:20]


class EmbeddingCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _paths(self, metadata: CacheMetadata) -> tuple[Path, Path]:
        stem = self.directory / metadata.key
        return stem.with_suffix(".pt"), stem.with_suffix(".json")

    def save(
        self, metadata: CacheMetadata, sample_ids: Sequence[str], embeddings: Tensor
    ) -> None:
        sample_ids = tuple(sample_ids)
        if len(sample_ids) != len(set(sample_ids)) or any(not value for value in sample_ids):
            raise ValueError("sample_ids must be non-empty and unique")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(sample_ids):
            raise ValueError("embeddings must be [samples, dimensions]")
        dtype = str(embeddings.dtype).removeprefix("torch.")
        if dtype != metadata.dtype:
            raise ValueError(f"embedding dtype {dtype!r} does not match cache metadata")
        self.directory.mkdir(parents=True, exist_ok=True)
        tensor_path, manifest_path = self._paths(metadata)
        tensor_tmp = tensor_path.with_suffix(".pt.tmp")
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        torch.save(embeddings.detach().cpu(), tensor_tmp)
        manifest_tmp.write_text(
            json.dumps({"metadata": asdict(metadata), "sample_ids": sample_ids}, indent=2),
            encoding="utf-8",
        )
        os.replace(tensor_tmp, tensor_path)
        os.replace(manifest_tmp, manifest_path)

    def load(
        self, metadata: CacheMetadata, sample_ids: Sequence[str] | None = None
    ) -> Tensor:
        tensor_path, manifest_path = self._paths(metadata)
        if not tensor_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"embedding cache miss for key {metadata.key}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("metadata") != asdict(metadata):
            raise ValueError("embedding cache metadata mismatch")
        stored_ids = tuple(manifest.get("sample_ids", ()))
        if len(stored_ids) != len(set(stored_ids)):
            raise ValueError("cached sample IDs are not unique")
        embeddings = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
            raise ValueError("cached embeddings are invalid")
        if embeddings.shape[0] != len(stored_ids):
            raise ValueError("cached embeddings and sample IDs are misaligned")
        dtype = str(embeddings.dtype).removeprefix("torch.")
        if dtype != metadata.dtype:
            raise ValueError("cached embedding dtype does not match metadata")
        if sample_ids is None:
            return embeddings
        index = {sample_id: row for row, sample_id in enumerate(stored_ids)}
        requested = tuple(sample_ids)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("requested sample IDs must be non-empty and unique")
        missing = [sample_id for sample_id in requested if sample_id not in index]
        if missing:
            raise ValueError(f"cache is missing sample IDs: {missing[:3]}")
        return embeddings[[index[sample_id] for sample_id in requested]]

    def get_or_compute(
        self,
        metadata: CacheMetadata,
        sample_ids: Sequence[str],
        compute: Callable[[], Tensor],
    ) -> tuple[Tensor, bool]:
        try:
            return self.load(metadata, sample_ids), True
        except FileNotFoundError:
            embeddings = compute()
            self.save(metadata, sample_ids, embeddings)
            return self.load(metadata, sample_ids), False
