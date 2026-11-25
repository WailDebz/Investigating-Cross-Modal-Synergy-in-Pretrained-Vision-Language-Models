# Phase 1.1 Plan: Data and Reproducibility Foundation

Branch: `phase1/1.1-data-reproducibility`  
Suggested owner: Student A  
Parent scope: Phase 1 only  
Mapped master milestones: 0, 1, 2, and 3

## 1. Mission

Phase 1.1 produces the trustworthy input side of the project. It preserves the completed TP1, establishes deterministic experiment identities and splits, and implements the extended image-text benchmark. Its final product is a tested data interface that Phase 1.2 can consume without knowing how samples were generated.

Phase 1.1 does not implement CLIP, neural fusion heads, lightweight adaptation, the unified training runner, or the final research experiments.

## 2. Owned files

Student A owns these paths until Phase 1.1 is merged:

```text
legacy/
src/synergy_bench/data/
tests/test_legacy_regression.py
tests/test_dataset_semantics.py
tests/test_reproducibility.py
tests/fixtures/data_contract/
configs/phase1/data/
docs/planning/phase-1.1-data-reproducibility.md
docs/decisions.md                 # data/oracle decisions only
```

Student A also creates the initial versions of:

```text
pyproject.toml
.gitignore
src/synergy_bench/__init__.py
src/synergy_bench/data/__init__.py
```

After that initial setup commit, Student B must not edit those shared files until Phase 1.1 has merged. Student B records needed model dependencies in the Phase 1.2 pull request description and adds them after rebasing.

Do not edit these Student B paths:

```text
src/synergy_bench/models/
src/synergy_bench/evaluation/
src/synergy_bench/cli.py
scripts/
tests/test_fusion_shapes.py
tests/test_cache_integrity.py
```

## 3. Interface contract with Phase 1.2

This contract should be agreed before implementation begins. If it must change later, both students record the change in `docs/decisions.md` before changing code.

### 3.1 Sample identity

Every sample must have a stable string `sample_id` derived from:

```text
benchmark_variant / scenario / data_seed / sample_index
```

Sample IDs must not depend on split, render chunk size, model, or training seed.

### 3.2 Dataset record

The model-facing dataset item returns this logical structure:

```python
{
    "sample_id": str,
    "image": Tensor,          # float32 [3, H, W] after declared model preprocessing
    "text": str,              # raw generated caption
    "target": Tensor,         # scalar int64, value 0 or 1
    "availability": {"image": bool, "text": bool},
    "metadata": {...},        # analysis only; never passed to the model
}
```

`metadata` may contain true latents, noisy observations, text category, template ID, and oracle prediction. The training runner must require an explicit debug/analysis flag to access it, preventing accidental privileged-feature leakage.

### 3.3 Split manifest

Each persisted manifest has one row per sample and at least these columns:

```text
sample_id, benchmark_variant, scenario, data_seed, split, target
```

Optional analysis columns belong in a separate latent manifest or clearly prefixed columns. Phase 1.2 joins embeddings and results by `sample_id`, never by row position alone.

### 3.4 Loader API

Phase 1.1 exposes a small public API with stable names:

```python
build_dataset(config) -> DatasetBundle
load_split(bundle, split_name) -> Dataset
load_manifest(config) -> DataFrame
make_nested_train_subsets(manifest, fractions, seed) -> dict[str, list[str]]
```

`DatasetBundle` exposes benchmark metadata, split manifests, and dataset objects but does not expose model logic.

### 3.5 Configuration keys

The data configuration must include:

```yaml
benchmark_variant: legacy_tp1 | vl_matched
scenario: A | B | C | D | E
n_samples: 5000
data_seed: 0
split_seed: 0
image:
  size: 224
  channels: 3
  render_chunk_size: 128
text:
  noise: 1.0
  template_set: v1
split:
  train: 0.6
  validation: 0.2
  test: 0.2
```

Model-specific preprocessing is owned by Phase 1.2. Phase 1.1 owns the fixed benchmark rendering transform and must expose its version in metadata.

## 4. Work packages

### 1.1-A - Freeze and organize the legacy TP1

Tasks:

1. Record the hashes of the current generator and notebook.
2. Move the generator and notebook together into `legacy/tp1/` without changing their contents.
3. Add a legacy README explaining how to open the notebook and what must remain unchanged.
4. Rename/move the two reference PDFs into `docs/references/` using descriptive names.
5. Preserve the planning instructions according to the team's versioning decision.

Acceptance criteria:

- Hashes of the moved TP1 files match their pre-move hashes.
- The notebook still resolves its relative `multimodal_gen` import.
- Git history clearly separates the mechanical move from later implementation.

Suggested commit:

```text
chore: preserve and organize legacy TP1 artifacts
```

### 1.1-B - Create the reproducibility foundation

Tasks:

1. Create the package skeleton, minimal dependency manifest, and ignore rules.
2. Implement named seed derivation for latent generation, split, image rendering, text templates, corruption, subset selection, model initialization, and training order.
3. Define stable sample IDs and persisted split manifests.
4. Implement nested, stratified training subsets with fixed validation/test samples.
5. Create a config loader with validation and a config hash.

Acceptance criteria:

- Repeating the same configuration produces byte-identical manifests.
- Changing a model seed cannot change sample IDs or splits.
- Training subsets are nested and class-balanced within a documented tolerance.
- Invalid split fractions or unsupported scenarios fail with clear errors.

Suggested commits:

```text
build: add minimal Phase 1 environment and package layout
feat(data): add named seeds and stable sample manifests
feat(data): add fixed splits and nested training subsets
```

### 1.1-C - Reproduce the legacy data path

Tasks:

1. Add a thin research-facing adapter around the legacy generator; do not rewrite it first.
2. Check A-E class balance, shapes, latent keys, and oracle outputs.
3. Store the notebook's reference metrics/tolerances for later end-to-end regression.
4. Document known differences between the historical notebook and the new protocol.

Acceptance criteria:

- Fixed legacy configurations reproduce generator arrays and oracle values.
- Scenario C is marginally balanced and each individual latent channel remains near chance.
- No legacy file has been silently edited.

Suggested commit:

```text
test(data): add legacy TP1 regression coverage
```

### 1.1-D - Implement `vl_matched`

Tasks:

1. Implement the approved image-text variants A-E.
2. Generate text from its assigned noisy observation, with independent template RNG.
3. Preserve image latent semantics at 16 and 224 px.
4. Render deterministically in chunks and use a fixed global uint8 mapping.
5. Expand grayscale to RGB without introducing an unintended color latent.
6. Retain true/noisy latents, emitted text category, and template ID as analysis metadata.

Acceptance criteria:

- Identical sample IDs render identically across runs and chunk sizes.
- Changing text template selection does not change images or other modalities.
- Changing image resolution preserves normalized blob position and grating orientation.
- Peak memory stays inside the predeclared smoke/full limit.

Suggested commits:

```text
feat(data): add controlled template-based text channel
feat(data): add deterministic chunked high-resolution rendering
feat(data): add matched vision-language A-E variants
```

### 1.1-E - Add text-aware oracles and leakage audits

Tasks:

1. Rename the legacy ceiling in research outputs to `latent_observation_oracle`.
2. Implement the approved text-aware oracle per scenario.
3. Audit class balance, unimodal predictability, joint oracle accuracy, template frequencies, caption lengths, and duplicates.
4. Add a simple bag-of-words/template shortcut diagnostic for Scenario C.

Acceptance criteria:

- Scenario C image-only and text-only latent predictors are approximately 0.5.
- The joint C oracle is above chance and responds monotonically to channel noise.
- Template ID and caption length do not predict the XOR target beyond the approved tolerance.
- Oracle inputs and limitations are written into dataset metadata.

Suggested commit:

```text
feat(data): add text-aware oracle and leakage audits
```

### 1.1-F - Publish the integration fixture

Tasks:

1. Export a tiny deterministic fixture covering A, C, and E with train/validation/test sample IDs.
2. Include expected shapes, targets, captions, availability flags, and manifest hashes.
3. Notify Student B of the exact branch commit containing the fixture.
4. Run the complete Phase 1.1 test suite before opening the merge request.

Acceptance criteria:

- Phase 1.2 can run model forward passes using only the fixture and the documented contract.
- The fixture is small enough to commit and contains no model-generated artifacts.
- Contract fields match the documentation exactly.

Suggested commit:

```text
test(data): publish deterministic model integration fixture
```

## 5. Branch workflow

Start only after the planning commit is present on `main`:

```bash
git switch main
git pull --ff-only
git switch -c phase1/1.1-data-reproducibility
git push -u origin phase1/1.1-data-reproducibility
```

Work in small, reviewable commits. Before every push:

```bash
git status
pytest tests/test_legacy_regression.py tests/test_dataset_semantics.py tests/test_reproducibility.py
git push
```

Keep the branch current without rewriting shared history:

```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
```

Use `--force-with-lease` only on your own feature branch, never on `main`, and tell Student B before rebasing if they are temporarily testing a commit from your branch.

## 6. Pull request checklist

- [ ] All owned tests pass from a clean environment.
- [ ] Legacy hashes and move are documented.
- [ ] The interface fixture and manifest hash are committed.
- [ ] Dataset/oracle decisions are recorded.
- [ ] No model, CLIP, fusion, or Phase 2 experiment code was added.
- [ ] Generated datasets, caches, and high-resolution images are not committed.
- [ ] Student B reviewed the public data contract.

Phase 1.1 merges into `main` before Phase 1.2. Once merged, Student A helps review the Phase 1.2 integration but does not start unplanned Phase 2 experiments.
