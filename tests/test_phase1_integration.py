from pathlib import Path

from synergy_bench.experiment import run_phase1
from synergy_bench.models.foundation import FrozenCLIP

from model_fakes import FakeCLIP, FakeProcessor


REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def _fake_clip(checkpoint: str, revision: str) -> FrozenCLIP:
    return FrozenCLIP(FakeCLIP(), FakeProcessor(), checkpoint, revision)


def test_all_families_share_the_phase11_manifest_and_reuse_cache(tmp_path: Path):
    data = tmp_path / "data.yaml"
    data.write_text(
        """benchmark_variant: vl_matched
scenario: C
n_samples: 30
data_seed: 0
split_seed: 0
image:
  size: 16
  channels: 3
  render_chunk_size: 8
text:
  noise: 1.0
  template_set: v1
split:
  train: 0.6
  validation: 0.2
  test: 0.2
""",
        encoding="utf-8",
    )
    config = tmp_path / "smoke.yaml"
    config.write_text(
        f"""version: 1
clip:
  checkpoint: fake/clip
  revision: {REVISION}
seeds: [0]
batch_size: 4
sample_limits:
  train: 8
  validation: 4
  test: 4
training:
  epochs: 1
  learning_rate: 0.001
  patience: 1
model:
  embedding_dim: 8
  hidden_dim: 8
  bilinear_projection_dim: 4
experiments:
  - data: data.yaml
    families: [from_scratch, frozen_clip, adapted_clip]
    modalities: [image, text]
    head: mlp
""",
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    raw = artifacts / "results" / "raw.jsonl"

    first = run_phase1(config, artifacts, raw, clip_factory=_fake_clip)
    second = run_phase1(config, artifacts, raw, clip_factory=_fake_clip)

    assert [result.family for result in first] == [
        "from_scratch",
        "frozen_clip",
        "adapted_clip",
    ]
    assert len({result.sample_manifest_hash for result in first}) == 1
    assert all(result.modalities == ("image", "text") for result in first)
    assert next(result for result in first if result.family == "frozen_clip").metrics[
        "cache_hit"
    ] == 0.0
    assert next(result for result in second if result.family == "frozen_clip").metrics[
        "cache_hit"
    ] == 1.0
    assert raw.is_file()
    assert len(list((artifacts / "manifests").glob("*.csv"))) == 1
    assert len(list((artifacts / "embeddings").glob("*.pt"))) == 2
