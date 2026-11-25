"""Run the Phase 1 model families against the shared data contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from torch import Tensor
from torch.utils.data import default_collate

from .data import build_dataset, load_split
from .data.manifest import manifest_to_csv_bytes
from .evaluation.cache import CacheMetadata, EmbeddingCache
from .evaluation.contracts import RunResult
from .evaluation.core import timed_call
from .evaluation.extraction import extract_to_cache
from .evaluation.reporting import write_results
from .evaluation.runner import ExperimentIdentity, evaluate_model
from .evaluation.training import fit_binary
from .models.adaptation import AdaptiveCLIPClassifier
from .models.contracts import ModelBatch, adapt_batch, seeded_model_init
from .models.foundation import FrozenCLIP
from .models.fusion import (
    BilinearFusionHead,
    EmbeddingClassifier,
    LinearFusionHead,
    MLPFusionHead,
)
from .models.scratch import ScratchClassifier, TextVocabulary, scratch_input_dims


ClipFactory = Callable[[str, str], FrozenCLIP]
_FAMILIES = {"from_scratch", "frozen_clip", "adapted_clip"}
_MODALITIES = {"image", "text"}


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _embedding_code_hash() -> str:
    root = Path(__file__).parent
    paths = (
        Path(__file__),
        root / "models" / "foundation.py",
        root / "evaluation" / "extraction.py",
    )
    return _sha256(b"".join(path.read_bytes() for path in paths))


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("experiment config must be a mapping")
    required = {
        "version",
        "clip",
        "seeds",
        "batch_size",
        "sample_limits",
        "training",
        "model",
        "experiments",
    }
    if set(values) != required:
        raise ValueError(f"experiment config fields must be exactly {sorted(required)}")
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return values, _sha256(canonical)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config["version"] != 1:
        raise ValueError("only experiment config version 1 is supported")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
    ):
        raise ValueError("seeds must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    limits = config["sample_limits"]
    if set(limits) != {"train", "validation", "test"} or any(
        value is not None and (not isinstance(value, int) or value <= 0)
        for value in limits.values()
    ):
        raise ValueError("sample limits must be positive integers or null")
    training = config["training"]
    if set(training) != {"epochs", "learning_rate", "patience"} or any(
        not isinstance(training[name], (int, float)) or training[name] <= 0
        for name in training
    ):
        raise ValueError("training requires positive epochs, learning_rate, and patience")
    model = config["model"]
    if set(model) != {"embedding_dim", "hidden_dim", "bilinear_projection_dim"} or any(
        not isinstance(value, int) or value <= 0 for value in model.values()
    ):
        raise ValueError("model dimensions must be positive integers")
    clip = config["clip"]
    if set(clip) != {"checkpoint", "revision"} or any(
        not isinstance(value, str) or not value or value == "main" or value.startswith("REPLACE_")
        for value in clip.values()
    ):
        raise ValueError("clip checkpoint and immutable revision are required")
    experiments = config["experiments"]
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a non-empty list")
    for experiment in experiments:
        if set(experiment) != {"data", "families", "modalities", "head"}:
            raise ValueError("each experiment requires data, families, modalities, and head")
        families = experiment["families"]
        modalities = experiment["modalities"]
        if not families or any(value not in _FAMILIES for value in families):
            raise ValueError(f"families must be selected from {sorted(_FAMILIES)}")
        if not modalities or any(value not in _MODALITIES for value in modalities):
            raise ValueError(f"modalities must be selected from {sorted(_MODALITIES)}")
        if len(set(families)) != len(families) or len(set(modalities)) != len(modalities):
            raise ValueError("families and modalities must be unique")
        if experiment["head"] not in {"linear", "mlp", "bilinear"}:
            raise ValueError("head must be linear, mlp, or bilinear")
        if experiment["head"] == "bilinear" and len(modalities) != 2:
            raise ValueError("bilinear heads require both modalities")


class _LazyBatches(Sequence[ModelBatch]):
    def __init__(self, dataset, batch_size: int, limit: int | None) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.count = len(dataset) if limit is None else min(limit, len(dataset))
        if self.count == 0:
            raise ValueError("every split must contain at least one selected sample")

    def __len__(self) -> int:
        return (self.count + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> ModelBatch:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.batch_size
        stop = min(start + self.batch_size, self.count)
        return adapt_batch(default_collate([self.dataset[row] for row in range(start, stop)]))

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self.dataset.sample_ids[: self.count])


class _CombinedBatches(Sequence[ModelBatch]):
    def __init__(self, parts: Sequence[Sequence[ModelBatch]]) -> None:
        self.parts = tuple(parts)

    def __len__(self) -> int:
        return sum(len(part) for part in self.parts)

    def __getitem__(self, index: int) -> ModelBatch:
        if index < 0:
            index += len(self)
        for part in self.parts:
            if index < len(part):
                return part[index]
            index -= len(part)
        raise IndexError(index)


def _batches(dataset, batch_size: int, limit: int | None) -> _LazyBatches:
    return _LazyBatches(dataset, batch_size, limit)


def _fusion(
    name: str,
    dimensions: Mapping[str, int],
    hidden_dim: int,
    projection_dim: int,
):
    if name == "linear":
        return LinearFusionHead(dimensions)
    if name == "mlp":
        return MLPFusionHead(dimensions, hidden_dim=hidden_dim)
    if name == "bilinear":
        return BilinearFusionHead(dimensions, projection_dim=projection_dim)
    raise ValueError(f"unsupported fusion head {name!r}")


def _embedding_batches(
    batches: Sequence[ModelBatch], embeddings: Mapping[str, Tensor]
) -> list[ModelBatch]:
    converted = []
    offset = 0
    for batch in batches:
        stop = offset + len(batch.sample_ids)
        converted.append(
            ModelBatch(
                sample_ids=batch.sample_ids,
                modalities={name: values[offset:stop] for name, values in embeddings.items()},
                targets=batch.targets,
                availability=batch.availability,
            )
        )
        offset = stop
    if any(offset != len(values) for values in embeddings.values()):
        raise ValueError("embedding rows do not match the selected dataset rows")
    return converted


def _identity(
    config_hash: str,
    manifest_hash: str,
    seed: int,
    family: str,
    modalities: Sequence[str],
    head: str,
) -> ExperimentIdentity:
    return ExperimentIdentity(
        config_hash=config_hash,
        sample_manifest_hash=manifest_hash,
        seed=seed,
        family=family,
        modalities=tuple(modalities),
        head=head,
    )


def _train(
    model,
    train_batches: Sequence[ModelBatch],
    validation_batches: Sequence[ModelBatch],
    training: Mapping[str, Any],
    seed: int,
):
    return fit_binary(
        model,
        train_batches,
        validation_batches,
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        patience=int(training["patience"]),
        seed=seed,
    )


def _default_clip_factory(checkpoint: str, revision: str) -> FrozenCLIP:
    return FrozenCLIP.from_pretrained(revision=revision, model_id=checkpoint)


def run_phase1(
    config_path: str | Path,
    artifacts: str | Path,
    results_path: str | Path,
    *,
    clip_factory: ClipFactory | None = None,
) -> list[RunResult]:
    """Run the configured Phase 1 matrix and write manifests plus raw result rows."""

    config_path = Path(config_path).resolve()
    artifacts = Path(artifacts)
    config, root_hash = _load_config(config_path)
    _validate_config(config)
    clip_factory = clip_factory or _default_clip_factory
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "configs").mkdir(exist_ok=True)
    (artifacts / "configs" / f"{root_hash}.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    results: list[RunResult] = []
    for experiment in config["experiments"]:
        data_path = (config_path.parent / experiment["data"]).resolve()
        bundle = build_dataset(data_path)
        manifest_hash = bundle.metadata["manifest_hash"]
        run_hash = _sha256(f"{root_hash}:{bundle.config_hash}")
        manifests = artifacts / "manifests"
        manifests.mkdir(exist_ok=True)
        (manifests / f"{bundle.config.scenario}-{manifest_hash[:16]}.csv").write_bytes(
            manifest_to_csv_bytes(bundle.manifest)
        )
        raw_batches = {
            split: _batches(
                load_split(bundle, split),
                config["batch_size"],
                config["sample_limits"][split],
            )
            for split in ("train", "validation", "test")
        }
        reference_ids = {split: batches.sample_ids for split, batches in raw_batches.items()}
        modalities = tuple(experiment["modalities"])
        model_config = config["model"]

        frozen_data = None
        if "frozen_clip" in experiment["families"]:
            encoder = clip_factory(config["clip"]["checkpoint"], config["clip"]["revision"])
            combined = _CombinedBatches(
                [raw_batches[split] for split in ("train", "validation", "test")]
            )
            cache = EmbeddingCache(artifacts / "embeddings")
            metadata = {
                modality: CacheMetadata(
                    sample_manifest_hash=manifest_hash,
                    checkpoint=encoder.model_id,
                    revision=encoder.revision,
                    preprocessing_hash=_sha256(
                        f"{encoder.model_id}:{encoder.revision}:float01:no-rescale:normalized"
                    ),
                    code_hash=_embedding_code_hash(),
                    config_hash=run_hash,
                    dtype="float32",
                    modality=modality,
                )
                for modality in ("image", "text")
            }
            (extracted, extraction_seconds) = timed_call(
                extract_to_cache, encoder, combined, cache, metadata
            )
            sample_ids, embeddings, cache_hit = extracted
            expected_ids = sum(
                (reference_ids[split] for split in ("train", "validation", "test")), ()
            )
            if sample_ids != expected_ids:
                raise ValueError("CLIP extraction changed sample order")
            all_embedding_batches = _embedding_batches(combined, embeddings)
            counts = [len(raw_batches[split]) for split in ("train", "validation", "test")]
            first, second = counts[0], counts[0] + counts[1]
            frozen_data = (
                encoder,
                {
                    "train": all_embedding_batches[:first],
                    "validation": all_embedding_batches[first:second],
                    "test": all_embedding_batches[second:],
                },
                extraction_seconds,
                cache_hit,
            )

        for seed in config["seeds"]:
            for family in experiment["families"]:
                identity = _identity(
                    run_hash, manifest_hash, seed, family, modalities, experiment["head"]
                )
                if family == "from_scratch":
                    texts = [
                        text
                        for batch in raw_batches["train"]
                        for text in batch.modalities["text"]
                    ]
                    vocabulary = TextVocabulary.from_texts(texts) if "text" in modalities else None
                    with seeded_model_init(seed):
                        fusion = _fusion(
                            experiment["head"],
                            scratch_input_dims(modalities, model_config["embedding_dim"]),
                            model_config["hidden_dim"],
                            model_config["bilinear_projection_dim"],
                        )
                        model = ScratchClassifier(
                            fusion, vocabulary, embedding_dim=model_config["embedding_dim"]
                        )
                    training = _train(
                        model, raw_batches["train"], raw_batches["validation"], config["training"], seed
                    )
                    results.append(
                        evaluate_model(model, raw_batches["test"], identity, training=training)
                    )
                elif family == "frozen_clip":
                    if frozen_data is None:
                        raise RuntimeError("frozen CLIP extraction was not initialized")
                    encoder, batches, extraction_seconds, cache_hit = frozen_data
                    dimensions = {
                        modality: batches["train"][0].modalities[modality].shape[1]
                        for modality in modalities
                    }
                    with seeded_model_init(seed):
                        fusion = _fusion(
                            experiment["head"],
                            dimensions,
                            model_config["hidden_dim"],
                            model_config["bilinear_projection_dim"],
                        )
                        model = EmbeddingClassifier(fusion, encoder=encoder)
                    training = _train(
                        model, batches["train"], batches["validation"], config["training"], seed
                    )
                    results.append(
                        evaluate_model(
                            model,
                            batches["test"],
                            identity,
                            training=training,
                            extraction_seconds=extraction_seconds,
                            extra_metrics={"cache_hit": float(cache_hit)},
                        )
                    )
                elif family == "adapted_clip":
                    encoder = clip_factory(
                        config["clip"]["checkpoint"], config["clip"]["revision"]
                    )
                    dimension = encoder.model.visual_projection.out_features
                    with seeded_model_init(seed):
                        fusion = _fusion(
                            experiment["head"],
                            {modality: dimension for modality in modalities},
                            model_config["hidden_dim"],
                            model_config["bilinear_projection_dim"],
                        )
                        model = AdaptiveCLIPClassifier(encoder.model, encoder.processor, fusion)
                    training = _train(
                        model, raw_batches["train"], raw_batches["validation"], config["training"], seed
                    )
                    results.append(
                        evaluate_model(model, raw_batches["test"], identity, training=training)
                    )
                else:  # guarded by config validation
                    raise ValueError(f"unsupported family {family!r}")

        for split, expected in reference_ids.items():
            if expected != raw_batches[split].sample_ids:
                raise ValueError("model integration changed selected sample IDs")

    write_results(results_path, results)
    return results
