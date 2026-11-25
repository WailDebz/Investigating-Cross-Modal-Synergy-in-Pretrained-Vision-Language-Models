"""Tiny deterministic integration fixture of the data contract (Phase 1.1-F).

For scenarios A, C and E of the reference ``vl_matched`` configs, the first two
samples (by sample index) of each split are exported to:

    images.npz     sample_ids (str) and images (uint8 grayscale, n x H x W), compressed
    contract.json  contract description, config hashes, manifest hashes and items

Regenerate from the repository root::

    python -m synergy_bench.data.fixture tests/fixtures/data_contract
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .dataset import build_dataset, load_split
from .manifest import SPLITS
from .render import RENDER_VERSION, UINT8_RANGE

REPO = Path(__file__).resolve().parents[3]
FIXTURE_SCENARIOS = ("A", "C", "E")
PER_SPLIT = 2
ITEM_KEYS = ["sample_id", "image", "text", "target", "availability", "metadata"]


def _config_path(scenario: str) -> Path:
    return REPO / "configs" / "phase1" / "data" / f"vl_matched_{scenario}.yaml"


def export_contract_fixture(out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_ids, images, scenarios = [], [], {}
    for scenario in FIXTURE_SCENARIOS:
        bundle = build_dataset(_config_path(scenario))
        items = []
        for split in SPLITS:
            ds = load_split(bundle, split, with_metadata=True)
            for k in range(PER_SPLIT):
                item = ds[k]
                items.append({
                    "sample_id": item["sample_id"],
                    "split": split,
                    "target": int(item["target"]),
                    "text": item["text"],
                    "availability": item["availability"],
                    "image_index": len(sample_ids),
                    "metadata": item["metadata"],
                })
                sample_ids.append(item["sample_id"])
                images.append(bundle.gray_image(int(ds.indices[k])))
        cfg = bundle.config
        scenarios[scenario] = {
            "config": dataclasses.asdict(cfg),
            "config_hash": bundle.config_hash,
            "manifest_hash": bundle.metadata["manifest_hash"],
            "split_counts": bundle.metadata["split_counts"],
            "oracle": bundle.metadata["oracle"],
            "items": items,
        }
    size = cfg.image.size
    contract = {
        "item_keys": ITEM_KEYS,
        "image": {
            "shape": [cfg.image.channels, size, size],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "from_fixture": "images.npz['images'][image_index] / 255, replicated to every channel",
        },
        "target": {"dtype": "int64", "shape": [], "values": [0, 1]},
        "metadata_policy": "item['metadata'] is {} unless load_split(..., with_metadata=True); "
                           "the fixture stores the analysis metadata for checks only, never as model input",
        "render_version": RENDER_VERSION,
        "uint8_range": list(UINT8_RANGE),
        "image_tolerance_uint8": 1,
        "scenarios": scenarios,
    }
    np.savez_compressed(out_dir / "images.npz", sample_ids=np.array(sample_ids), images=np.stack(images))
    (out_dir / "contract.json").write_bytes((json.dumps(contract, indent=1, sort_keys=True) + "\n").encode("utf-8"))


def load_fixture_items(fixture_dir: str | Path, scenario: str, with_metadata: bool = False) -> list[dict]:
    """Contract items of one scenario rebuilt from the fixture alone (no generator needed)."""
    fixture_dir = Path(fixture_dir)
    contract = json.loads((fixture_dir / "contract.json").read_text(encoding="utf-8"))
    with np.load(fixture_dir / "images.npz") as npz:
        images, ids = npz["images"], npz["sample_ids"]
    channels = contract["image"]["shape"][0]
    out = []
    for rec in contract["scenarios"][scenario]["items"]:
        assert ids[rec["image_index"]] == rec["sample_id"]
        gray = torch.from_numpy(images[rec["image_index"]].astype(np.float32) / 255.0)
        out.append({
            "sample_id": rec["sample_id"],
            "image": gray.unsqueeze(0).repeat(channels, 1, 1),
            "text": rec["text"],
            "target": torch.tensor(rec["target"], dtype=torch.int64),
            "availability": dict(rec["availability"]),
            "metadata": dict(rec["metadata"]) if with_metadata else {},
        })
    return out


if __name__ == "__main__":
    export_contract_fixture(sys.argv[1] if len(sys.argv) > 1 else REPO / "tests" / "fixtures" / "data_contract")
