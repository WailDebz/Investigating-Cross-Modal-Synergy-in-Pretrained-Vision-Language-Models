# Phase 1.2 Plan: Models, Fusion, and Unified Evaluation

Branch: `phase1/1.2-models-evaluation`  
Suggested owner: Student B  
Parent scope: Phase 1 only  
Mapped master milestones: 4, 5, 6, 7, and 8

## 1. Mission

Phase 1.2 implements the model and evaluation side of the common foundation: from-scratch encoders, frozen CLIP encoders, one lightweight adaptation method, shared fusion heads, cost logging, aggregation, and one-command reproduction.

It begins in parallel using a tiny contract fixture. It must not copy the generator or invent a second data representation. After Phase 1.1 merges, this branch rebases onto `main`, connects to the real loader, and runs the integration suite.

Phase 1.2 does not run the full H1-H3 experiments or interpret Phase 1 diagnostics as Phase 2 research conclusions.

## 2. Owned files

Student B owns these paths:

```text
src/synergy_bench/models/
src/synergy_bench/evaluation/
src/synergy_bench/cli.py
tests/test_fusion_shapes.py
tests/test_cache_integrity.py
tests/test_model_families.py
tests/test_evaluation_protocol.py
tests/fixtures/model_contract/
configs/phase1/models/
scripts/reproduce_phase1.sh
docs/planning/phase-1.2-models-evaluation.md
```

Do not edit these Student A paths before Phase 1.1 merges:

```text
legacy/
src/synergy_bench/data/
tests/test_legacy_regression.py
tests/test_dataset_semantics.py
tests/test_reproducibility.py
tests/fixtures/data_contract/
configs/phase1/data/
```

Before Phase 1.1 merges, also avoid editing `pyproject.toml`, `.gitignore`, root package initializers, and final smoke/full configs. Keep a dependency checklist in the pull request description. Add those dependencies in one integration commit after rebasing onto the merged Phase 1.1 branch.

## 3. How Phase 1.2 starts without the final dataset

Create a small in-memory contract fixture that matches the documented Phase 1.1 item format:

```python
{
    "sample_id": str,
    "image": Tensor,          # [3, 224, 224]
    "text": str,
    "target": Tensor,
    "availability": {"image": bool, "text": bool},
    "metadata": {},
}
```

The fixture exists only to test batching, forward passes, fusion shapes, training loops, cache joins, and result schemas. It must not become an alternative generator. Delete or retain it strictly as a unit-test fixture after real integration.

All joins use `sample_id`. No evaluation component may assume that embedding row order is automatically aligned with label row order.

## 4. Work packages

### 1.2-A - Define common model and result contracts

Tasks:

1. Define modality-encoder and fusion-head interfaces.
2. Define a batch adapter that consumes only public sample fields.
3. Define tidy run/result records with configuration hash, sample-manifest hash, seed, family, modality set, head, metrics, parameters, timing, and status.
4. Implement total/trainable parameter counting and trainable-name reporting.
5. Add device-safe inference and deterministic model initialization hooks.

Acceptance criteria:

- A mock image encoder and text encoder can be swapped without changing fusion code.
- Result rows reject missing experiment identity fields.
- Metadata/latents never enter a model batch.
- CPU smoke forward and backward passes are deterministic under the same model seed.

Suggested commits:

```text
feat(models): define encoder and fusion interfaces
feat(eval): define reproducible run and result records
```

### 1.2-B - Implement shared fusion heads

Tasks:

1. Port and test concatenation.
2. Implement linear concatenation and concatenation plus MLP.
3. Port projected bilinear fusion with explicit parameter/activation checks.
4. Implement calibrated late fusion without using the test set.
5. Support availability masks consistently across comparable heads.

Acceptance criteria:

- Shape tests cover image-only, text-only, and image-text batches.
- Bilinear output size equals `(projection_dim + 1) ** 2` for two modalities.
- Missing-modality information is applied consistently or marked as a different protocol.
- Late-fusion calibration consumes training/out-of-fold or designated meta-training predictions, not test labels.

Suggested commit:

```text
feat(models): add shared linear nonlinear bilinear and late fusion heads
```

### 1.2-C - Implement the from-scratch family

Tasks:

1. Add a lightweight 224 px image encoder with strided convolution and adaptive pooling.
2. Add a small text encoder using trainable token embeddings, masked mean pooling, and an MLP projection.
3. Add unimodal and multimodal training paths.
4. Add validation checkpointing, fixed stopping rules, and timing.

Acceptance criteria:

- Image-only, text-only, linear, MLP, bilinear, and late-fusion smoke runs complete on CPU.
- A synthetic XOR embedding fixture is solved by MLP/bilinear heads while a linear head remains near chance.
- Every run records initialization seed, total/trainable parameters, epoch count, and elapsed time.

Suggested commit:

```text
feat(models): add from-scratch image and text family
```

### 1.2-D - Implement frozen CLIP and the embedding cache

Tasks:

1. Integrate one pinned CLIP ViT-B/32 checkpoint and processor.
2. Freeze encoders and extract image/text embeddings in batches.
3. Cache embeddings by sample ID with complete metadata.
4. Reject caches whose dataset manifest, checkpoint, preprocessing, code/config hash, or dtype differs.
5. Add zero-shot classification only for scenarios declared applicable.

Acceptance criteria:

- Frozen encoder trainable parameter count is zero.
- A second extraction request reuses a matching cache.
- Shuffled cache rows are correctly realigned by sample ID or fail clearly.
- Cache mismatch never causes silent reuse.
- No model download or cache artifact is committed to Git.

Suggested commits:

```text
feat(models): add pinned frozen CLIP encoders
feat(cache): add sample-aligned embedding cache with integrity checks
```

### 1.2-E - Implement lightweight adaptation

Tasks:

1. Freeze all CLIP parameters by default.
2. Unfreeze only the approved final transformer block and projection in each encoder.
3. Train on images/text in small batches; do not load cached final embeddings as training inputs.
4. Log trainable names, gradient presence, changed weights, memory, and time.

Acceptance criteria:

- Only approved encoder parameters and the head require gradients.
- At least one approved encoder parameter changes after a training step.
- Frozen-prefix weights remain identical.
- The adaptation smoke run completes within the approved resource budget.

Suggested commit:

```text
feat(models): add final-block CLIP adaptation family
```

### 1.2-F - Build the unified evaluator

Tasks:

1. Train/select configurations using validation data only.
2. Implement accuracy, optional balanced accuracy/NLL, oracle reporting, and per-seed output.
3. Implement nested sample-efficiency evaluation using the sample IDs supplied by Phase 1.1.
4. Implement a consistent missing/corrupted-modality protocol.
5. Record extraction/training/inference time, total/trainable parameters, peak memory, hardware, and versions.
6. Aggregate mean and standard deviation without discarding raw per-seed rows.

Acceptance criteria:

- Test labels are unavailable to selection code.
- Every comparison records identical dataset manifest hashes and modality sets.
- Re-aggregation from saved raw rows reproduces tables exactly without training.
- Three-seed outputs include individual rows and mean +/- standard deviation.

Suggested commits:

```text
feat(eval): add validation-only training and locked test evaluation
feat(eval): add nested sample efficiency corruption and cost logging
feat(eval): add deterministic aggregation tables and figures
```

### 1.2-G - Integrate Phase 1.1

This work starts only after Phase 1.1 is merged into `main`.

Tasks:

1. Rebase the Phase 1.2 branch onto updated `main`.
2. Replace the temporary loader fixture in integration tests with `build_dataset()` and `load_split()`.
3. Validate sample IDs, target alignment, preprocessing ownership, and manifest hashes.
4. Add model dependencies to the merged environment manifest.
5. Add final `smoke.yaml` and `full.yaml` by composing data and model settings.
6. Run the minimum full integration matrix defined in the master plan.

Acceptance criteria:

- No duplicated generator, split, or caption logic exists in model/evaluation modules.
- Model results reference the exact Phase 1.1 manifest hash.
- The full smoke suite passes from a fresh checkout.
- Scratch, frozen, and adapted families consume the same image-text sample IDs.

Suggested commit:

```text
integrate: connect model pipeline to Phase 1 data contract
```

### 1.2-H - Reproduction and documentation

Tasks:

1. Implement the CLI and `scripts/reproduce_phase1.sh`.
2. Update the root README with setup, smoke/full commands, artifact tree, and expected runtime.
3. Generate figures/tables only from raw result rows.
4. Document zero-shot applicability, oracle terminology, limitations, and AI-use policy.

Acceptance criteria:

- One smoke command creates manifests, results, tables, and figures in documented locations.
- A cache-enabled rerun is deterministic and faster.
- Phase 2 hypotheses appear only as future context, never as Phase 1 conclusions.

Suggested commits:

```text
feat(cli): add one-command Phase 1 reproduction
docs: add Phase 1 setup methods and limitations
```

## 5. Branch workflow

Start from the same planning commit as Phase 1.1:

```bash
git switch main
git pull --ff-only
git switch -c phase1/1.2-models-evaluation
git push -u origin phase1/1.2-models-evaluation
```

During parallel development, use only the model contract fixture. Regularly inspect Phase 1.1 without merging it into this branch:

```bash
git fetch origin
git log --oneline origin/phase1/1.1-data-reproducibility -10
```

After Phase 1.1 merges:

```bash
git fetch origin
git rebase origin/main
pytest
git push --force-with-lease
```

Resolve integration conflicts in favor of the merged Phase 1.1 data contract. Do not recreate deleted/moved legacy files or overwrite its manifests and tests.

## 6. Pull request checklist

- [ ] Branch is rebased onto `main` after Phase 1.1 merged.
- [ ] Scratch, frozen, and adapted families pass smoke tests.
- [ ] Frozen-cache integrity and sample alignment tests pass.
- [ ] Validation/test separation is tested.
- [ ] One-command smoke reproduction succeeds from a fresh checkout.
- [ ] No generated cache, checkpoint, dataset, image, or full result artifact is committed.
- [ ] No Phase 2 experiment or H1-H3 conclusion is included.
- [ ] Student A reviewed data-contract use and sample alignment.

Phase 1.2 merges only after Phase 1.1. After both branches are merged, tag the Phase 1 foundation only when the clean smoke reproduction and required tests pass on `main`.

## 7. Joint integration responsibility

Although Student B owns the integration commit, both students verify it:

- Student A checks dataset semantics, manifests, oracles, and absence of leakage.
- Student B checks model freezing/adaptation, fusion behavior, evaluation isolation, and cost logs.
- Both verify that direct comparisons use the same benchmark variant, modality set, sample IDs, and splits.
- Both approve the final Phase 1 tag and document their contributions.

## 8. Implementation status on `phase1.2/models-eval`

- [x] 1.2-A - common model and result contracts.
- [x] 1.2-B - shared linear, MLP, bilinear, and calibrated late-fusion heads.
- [x] 1.2-C - lightweight from-scratch image/text family and validation training loop.
- [x] 1.2-D - frozen CLIP wrapper, zero-shot utility, and sample-aligned embedding cache.
- [x] 1.2-E - final-block/projection adaptation wrapper operating on raw batches.
- [x] 1.2-F - metrics, validation-only selection, nested-subset checks, missingness, aggregation, costs, tables, and figures.
- [x] 1.2-G - integrated with the Phase 1.1 loaders, manifest hashes, environment, and composed smoke/full configurations.
- [x] 1.2-H - one-command training/report reproduction, root README, artifact layout, limitations, and AI-use declaration.
