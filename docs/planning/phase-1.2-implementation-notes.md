# Phase 1.2 Implementation Notes

## Implemented scope

- Common model, batch, parameter, and result contracts.
- Linear, MLP, exact two-modality bilinear, and calibrated late-fusion heads.
- Lightweight from-scratch image and text encoders.
- Frozen CLIP extraction with an immutable revision requirement.
- Sample-aligned embedding caches with metadata integrity checks.
- Final-block plus projection CLIP adaptation on raw batches.
- Validation checkpointing, locked-test selection helpers, nested-subset validation, metrics, cost fields, aggregation, CSV tables, and SVG figures.
- A reporting CLI and reproduction script for committed/raw result rows.

## Integrated environment

The merged environment manifest includes compatible versions of:

- PyTorch;
- Transformers;
- the CLIP tokenizer/image processor dependencies required by the pinned Transformers release.

No plotting dependency is required; Phase 1.2 writes a small SVG directly from aggregated raw rows.

## Phase 1.1 integration

Phase 1.1 provides:

- `build_dataset(config)` and `load_split(bundle, split_name)`;
- stable sample IDs and split manifests;
- `vl_matched` image-text samples following the documented public batch contract;
- the final smoke/full data configurations and their manifest hashes.

The integration runner now consumes these loaders directly, writes the exact manifest,
and runs all three model families on identical sample IDs. The committed integration
test uses a CLIP test double; `scripts/reproduce_phase1.sh smoke` additionally exercises
the pinned real checkpoint.

## Scientific reporting rules

- Report the legacy ceiling as `latent_observation_oracle`.
- A text-aware oracle comes only from Phase 1.1; Phase 1.2 does not reconstruct it.
- Zero-shot CLIP is reported only for scenarios marked applicable in `zero_shot_applicability.json`.
- Validation selects configurations. Test labels are used only once through the locked-test evaluation step.
- Frozen final embeddings are never used for the adaptation family.
- Phase 1 diagnostics are not presented as tests of H1-H3.

## Reproduction command

For a complete smoke run:

```bash
scripts/reproduce_phase1.sh smoke
```

This runs tests and training, then regenerates `artifacts/tables/summary.csv` and
`artifacts/figures/accuracy.svg` from `artifacts/results/raw.jsonl`. Use `full` only on
appropriate compute; its three-seed 5,000-sample matrix is intentionally much larger.
