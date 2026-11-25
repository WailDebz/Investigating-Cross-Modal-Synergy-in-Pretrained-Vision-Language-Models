"""Data side of the benchmark (Phase 1.1).

Public API (plan section 3.4):
build_dataset, load_split, load_manifest, make_nested_train_subsets.
"""
from .config import DataConfig, config_hash, load_data_config
from .dataset import DatasetBundle, build_dataset, load_manifest, load_split
from .splits import make_nested_train_subsets

__all__ = [
    "DataConfig",
    "DatasetBundle",
    "build_dataset",
    "config_hash",
    "load_data_config",
    "load_manifest",
    "load_split",
    "make_nested_train_subsets",
]
