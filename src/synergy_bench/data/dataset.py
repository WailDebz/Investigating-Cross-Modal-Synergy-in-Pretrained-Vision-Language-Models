"""Public loader API (plan section 3.4) and the model-facing dataset record (section 3.2).

Items::

    {"sample_id": str,
     "image": float32 tensor [C, H, W] = uint8 render / 255, grayscale replicated to C channels,
     "text": str,
     "target": int64 scalar tensor,
     "availability": {"image": True, "text": True},
     "metadata": {} unless load_split(..., with_metadata=True)}

Images are rendered on demand from the per-sample stream, or read from an
optional uint8 grayscale memmap cache (``build_dataset(config, cache_dir=...)``).
The full dataset is never materialized as float32 RGB.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig, config_hash, config_to_canonical_json, load_data_config
from .legacy import legacy_manifest
from .manifest import SPLITS, manifest_hash, manifest_to_csv_bytes
from .oracle import ORACLE_INFO
from .render import RENDER_VERSION, UINT8_RANGE, render_chunked, render_uint8
from .splits import make_split_manifest
from .text import template_set
from .vl_matched import channel_sigmas, generate_latents, pixel_noise_std


@dataclass
class DatasetBundle:
    """Benchmark metadata, split manifest and latent (analysis) manifest of one config."""

    config: DataConfig
    config_hash: str
    manifest: pd.DataFrame
    latents: pd.DataFrame
    metadata: dict
    image_cache: np.ndarray | None = None  # uint8 (n, H, W) memmap, or None for on-demand rendering

    def gray_image(self, index: int) -> np.ndarray:
        """uint8 (H, W) grayscale render of sample ``index``."""
        if self.image_cache is not None:
            return np.asarray(self.image_cache[index])
        row = self.latents.iloc[index]
        cfg = self.config
        return render_uint8(row.obs_img, row.nuisance, pixel_noise_std(cfg.scenario), cfg.image.size,
                            cfg.data_seed, index)


def _as_config(config) -> DataConfig:
    return config if isinstance(config, DataConfig) else load_data_config(config)


def _fractions(cfg: DataConfig) -> dict[str, float]:
    return {"train": cfg.split.train, "validation": cfg.split.validation, "test": cfg.split.test}


def _require_vl_matched(cfg: DataConfig) -> None:
    if cfg.benchmark_variant != "vl_matched":
        raise ValueError("build_dataset supports benchmark_variant 'vl_matched'; "
                         "use synergy_bench.data.legacy for 'legacy_tp1'")
    template_set(cfg.text.template_set)


def _latents_and_manifest(cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    latents, y = generate_latents(cfg.scenario, cfg.n_samples, cfg.data_seed, cfg.text.noise, cfg.text.template_set)
    manifest = make_split_manifest("vl_matched", cfg.scenario, cfg.data_seed, y, _fractions(cfg), cfg.split_seed)
    return latents, manifest


def load_manifest(config) -> pd.DataFrame:
    """Split manifest (contract columns) for a config or a YAML path; no image is rendered."""
    cfg = _as_config(config)
    if cfg.benchmark_variant == "legacy_tp1":
        return legacy_manifest(cfg)
    _require_vl_matched(cfg)
    return _latents_and_manifest(cfg)[1]


def _metadata(cfg: DataConfig, manifest: pd.DataFrame, latents: pd.DataFrame) -> dict:
    s_img, s_txt = channel_sigmas(cfg.scenario, cfg.text.noise)
    correct = latents.oracle_pred.to_numpy() == manifest.target.to_numpy()
    ts = template_set(cfg.text.template_set)
    return {
        "benchmark_variant": cfg.benchmark_variant,
        "scenario": cfg.scenario,
        "n_samples": cfg.n_samples,
        "config_hash": config_hash(cfg),
        "manifest_hash": manifest_hash(manifest),
        "split_counts": {s: int((manifest.split == s).sum()) for s in SPLITS},
        "render": {
            "version": RENDER_VERSION,
            "size": cfg.image.size,
            "channels": cfg.image.channels,
            "uint8_range": list(UINT8_RANGE),
            "image_tensor": "float32 = uint8 / 255, grayscale replicated to all channels",
            "pixel_noise_std": pixel_noise_std(cfg.scenario),
        },
        "text": {
            "template_set": cfg.text.template_set,
            "words": list(ts["words"]),
            "frames": list(ts["frames"]),
            "category_rule": "text_category = 1[obs_txt > 0]",
        },
        "sigma_img": s_img,
        "sigma_txt": s_txt,
        "oracle": {**ORACLE_INFO,
                   "accuracy": {s: float(correct[(manifest.split == s).to_numpy()].mean()) for s in SPLITS}},
        "latent_columns": list(latents.columns),
    }


def _open_or_build_cache(cfg: DataConfig, latents: pd.DataFrame, cache_dir: Path, digest: str) -> np.ndarray:
    size, n = cfg.image.size, cfg.n_samples
    path = cache_dir / f"images_{RENDER_VERSION}_{size}px_{digest[:16]}.npy"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, size, size))
        render_chunked(latents.obs_img.to_numpy(), latents.nuisance.to_numpy(), pixel_noise_std(cfg.scenario),
                       size, cfg.data_seed, np.arange(n), cfg.image.render_chunk_size, out=out)
        out.flush()
        del out
        os.replace(tmp, path)  # a partial cache is never visible under the final name
    cache = np.load(path, mmap_mode="r")
    if cache.shape != (n, size, size) or cache.dtype != np.uint8:
        raise RuntimeError(f"image cache {path} has {cache.dtype} {cache.shape}, expected uint8 {(n, size, size)}")
    return cache


def build_dataset(config, cache_dir: str | Path | None = None) -> DatasetBundle:
    """Build the ``vl_matched`` bundle for a config (or YAML path).

    With ``cache_dir`` (under ``artifacts/``), images are rendered once into a uint8
    memmap keyed by the config hash, and the manifest, latent manifest and metadata
    are written next to it. Without it, images are rendered on demand.
    """
    cfg = _as_config(config)
    _require_vl_matched(cfg)
    latents, manifest = _latents_and_manifest(cfg)
    digest = config_hash(cfg)
    metadata = _metadata(cfg, manifest, latents)
    cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir) / f"vl_matched_{cfg.scenario}_seed{cfg.data_seed}_{digest[:16]}"
        cache = _open_or_build_cache(cfg, latents, cache_dir, digest)
        (cache_dir / "config.json").write_bytes((config_to_canonical_json(cfg) + "\n").encode("utf-8"))
        (cache_dir / "manifest.csv").write_bytes(manifest_to_csv_bytes(manifest))
        (cache_dir / "latents.csv").write_bytes(latents.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        (cache_dir / "metadata.json").write_bytes((json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return DatasetBundle(cfg, digest, manifest, latents, metadata, cache)


class VLMatchedDataset(Dataset):
    """Torch dataset over one split; rows ordered by sample index."""

    def __init__(self, bundle: DatasetBundle, split_name: str, with_metadata: bool = False):
        if split_name not in SPLITS:
            raise ValueError(f"unknown split {split_name!r}; expected one of {SPLITS}")
        self.bundle = bundle
        self.split = split_name
        self.with_metadata = with_metadata
        self.indices = np.flatnonzero(bundle.manifest.split.to_numpy() == split_name)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def sample_ids(self) -> list[str]:
        return self.bundle.manifest.sample_id.to_numpy()[self.indices].tolist()

    def __getitem__(self, k: int) -> dict:
        i = int(self.indices[k])
        b = self.bundle
        gray = torch.from_numpy(b.gray_image(i).astype(np.float32) / 255.0)
        row = b.latents.iloc[i]
        item = {
            "sample_id": b.manifest.sample_id.iat[i],
            "image": gray.unsqueeze(0).repeat(b.config.image.channels, 1, 1),
            "text": row.text,
            "target": torch.tensor(int(b.manifest.target.iat[i]), dtype=torch.int64),
            "availability": {"image": True, "text": True},
            "metadata": {},
        }
        if self.with_metadata:
            meta = {k: v.item() if hasattr(v, "item") else v for k, v in row.items() if k not in ("sample_id", "text")}
            meta.update(split=self.split, render_version=RENDER_VERSION, template_set=b.config.text.template_set)
            item["metadata"] = meta
        return item


def load_split(bundle: DatasetBundle, split_name: str, with_metadata: bool = False) -> VLMatchedDataset:
    """Dataset of one split. Analysis metadata is returned only with ``with_metadata=True``."""
    return VLMatchedDataset(bundle, split_name, with_metadata)
