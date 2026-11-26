# Investigating Cross-Modal Synergy in Vision-Language Models

Phase 1 provides a reproducible extension of the original TP1 benchmark. It compares
from-scratch image/text encoders, frozen CLIP embeddings, and final-block CLIP
adaptation on identical samples and splits. It does not claim results for the Phase 2
H1-H3 research hypotheses.

## Setup

Python 3.10 or newer is required. Python 3.11 is the tested version.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The first foundation-model run downloads the pinned CLIP ViT-B/32 revision. Model
weights and every generated artifact are ignored by Git.

## Reproduce Phase 1

Run the complete smoke workflow:

```bash
scripts/reproduce_phase1.sh smoke
```

This runs the test suite, builds the Phase 1.1 manifest, executes scratch, frozen, and
adapted families on the same Scenario C sample IDs, writes raw results, and regenerates
the table and figure. The smoke configuration uses one seed, one epoch, eight training
samples, four validation samples, and four locked-test samples. It validates the
pipeline; its metrics are not research conclusions. A cached rerun avoids CLIP feature
extraction and should be faster.

The predeclared full matrix is available for suitable compute:

```bash
scripts/reproduce_phase1.sh full
```

It uses three seeds and full 5,000-sample configurations. Run it only after reviewing
the expected CPU/GPU time and storage cost.

Generated files are written under:

```text
artifacts/
├── configs/       # exact experiment snapshots
├── embeddings/    # sample-aligned frozen CLIP cache
├── figures/       # accuracy.svg
├── manifests/     # Phase 1.1 split manifests
├── results/       # append-ready raw result records
└── tables/        # summary.csv
```

## Scientific boundaries

- `latent_observation_oracle` names the historical TP1 ceiling.
- `text_aware_oracle` is the Phase 1.1 image-text oracle; its inputs and optimistic
  image-observation limitation are recorded in dataset metadata.
- Validation data selects checkpoints. Test labels are used only for final evaluation.
- Frozen CLIP consumes cached final embeddings. Adapted CLIP consumes raw images and
  text and updates only the approved final blocks and projections.
- Zero-shot CLIP is applicable only to scenarios declared in
  `configs/phase1/models/zero_shot_applicability.json`.
- The smoke workflow is an engineering check, not evidence for Phase 2 hypotheses.
