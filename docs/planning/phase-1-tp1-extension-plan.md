# Phase 1 Plan: Research-Grade Extension of TP1

Status: proposed for discussion and approval  
Scope: Phase 1 only - the common experimental foundation and TP1 extension  
Primary track context: Track 1 (Synergy)  
Out of scope: the full H1-H3 hypothesis experiments and all Phase 2 conclusions

## 1. Executive decision summary

Phase 1 should preserve the completed TP1 as an immutable reference and build a separate, reproducible research pipeline around it. The main comparison must use the same image-text samples, splits, fusion heads, and evaluation rules for all three required model families. Results from the original image-signal-tabular benchmark and results from the new matched vision-language benchmark must never be mixed into a direct encoder comparison.

Recommended defaults, subject to approval:

1. Keep an exact `legacy_tp1` benchmark and introduce a separately named `vl_matched` variant.
2. Make Scenario C in `vl_matched` the central Phase 1 validation task: the image carries one noisy bit, text carries an independent noisy bit, and the label is their XOR.
3. Generate text from a text-channel noisy observation, not from the label and not directly from all true latents.
4. Use deterministic 224 x 224 rendering, fixed grayscale-to-RGB mapping, and chunked/on-demand generation rather than materializing float32 RGB images.
5. Start with one foundation model, CLIP ViT-B/32, and cache frozen image/text embeddings with complete cache metadata.
6. Implement lightweight adaptation by unfreezing only the final transformer block and projection of each CLIP encoder. This is simpler than LoRA for the first working adaptation family and is genuinely different from head-only training.
7. Use three independent experiment seeds, nested training subsets, a fixed validation/test split within each seed, validation-only model selection, and one final test evaluation.
8. Report the existing latent-observation oracle under that precise name. Add a text-aware oracle for the extended generative process; do not silently reuse the old oracle.

The professor's brief makes depth more important than breadth. Phase 1 should establish a trustworthy pipeline; it should not attempt multiple foundation models, all possible adaptation methods, or the complete H1-H3 research matrix.

## 2. Repository audit

### 2.1 Existing artifacts and relationships

| Artifact | Role | Audit finding | Phase 1 treatment |
|---|---|---|---|
| `multimodal_gen.py` | Original synthetic generator | Implements A-E, latent observations, an optimistic oracle, splitting, corruption, handcrafted features, and plotting. `img_size` is already configurable, but rendering is fully vectorized and grayscale. | Preserve unchanged as the TP1 reference. Reuse tested logic through a new research generator rather than silently changing legacy behavior. |
| `multimodal_fusion_lab_student.ipynb` | Completed TP1 notebook/report-like record | Contains completed exercises, stored single-run outputs, classical baselines, learned encoders, four fusion modules, modality dropout, pretrain-then-fuse, corruption, sweeps, and a conflict scenario. It imports `multimodal_gen.py` from the same directory. | Preserve notebook and outputs. Treat its numerical claims as historical single-seed observations until reproduced. |
| `Evaluation project when does a multimodal foundation model actually help.pdf` | Professor's specification | Authoritative requirements: text, high-resolution images, three model families, zero-shot where meaningful, common protocol, compute constraints, and deliverables. | Retain as an authoritative reference and map every requirement to a test or output. |
| `projet_mahb (1).pdf` | Track 1 proposal | Defines the research question and H1-H3. H1-H3 motivate architecture hooks but belong to Phase 2. | Retain as research context. Do not execute the full planned experiments in Phase 1. |
| `instructions.md` | Local planning brief | Explicitly limits the current mission to planning and asks for this audit, matrix, architecture, roadmap, feasibility analysis, and approval questions. It is currently ignored by Git. | Keep as local instructions unless the team explicitly decides to version it. |
| `README.md` | Repository entry point | Contains only the project title. | Replace after plan approval with setup, scope, commands, artifact map, and reproducibility instructions. |
| `.gitignore` | Ignore rules | Currently contains only `instructions.md`; it has no entries for caches, datasets, checkpoints, results, or notebook/runtime files. | Expand after plan approval, preserving the intentional treatment of `instructions.md`. |

### 2.2 What is already reusable

- Scenario definitions A-E and most latent-variable logic.
- The fixed tabular world matrix across splits.
- Corruption operators and modality-level evaluation ideas.
- Lightweight from-scratch encoders as initial baselines.
- Concatenation, gated fusion, exact tensor/bilinear fusion, late averaging, and nonlinear stacking concepts.
- Modality dropout, permutation importance, and pretrain-then-fuse as optional diagnostic baselines.
- The existing stored notebook outputs as a regression reference, not as final research results.

### 2.3 What is missing

- A standalone TP1 report is not present. The notebook has narrative answers, but it is not a separate report artifact.
- No dependency/environment manifest, automated tests, configuration system, command-line runner, result schema, split manifests, or machine-readable experiment outputs exist.
- No text modality, text encoder, CLIP integration, embedding cache, lightweight encoder adaptation, or zero-shot baseline exists.
- No systematic validation protocol, multi-seed aggregation, trainable-parameter reporting, extraction timing, or training timing exists.
- No one-command reproduction path exists.

## 3. Requirements traceability matrix

Legend: implemented means present in TP1; partial means useful code exists but is not research-ready; missing means Phase 1 must add it.

| Requirement source | Requirement | Current status | Phase 1 deliverable / acceptance evidence |
|---|---|---:|---|
| Professor | Preserve A-E scenario logic | Implemented | Legacy regression tests plus separately named extended variants. |
| Professor | Add template-based text tied to some, not all, latents | Missing | Deterministic caption generator, leakage audit, text-channel metadata, scenario tests. |
| Professor | High-resolution images compatible with pretrained encoders | Partial | Configurable deterministic 224 px renderer, RGB conversion policy, preprocessing metadata, memory test. |
| Professor | From-scratch family | Partial | Reusable image/text encoders and matched fusion heads on `vl_matched`. |
| Professor | Frozen foundation encoders and cached embeddings | Missing | CLIP extractor, immutable cache key/manifest, cache reuse test. |
| Professor | Lightweight adaptation | Missing | Final-block adaptation pipeline showing encoder parameters change and final embeddings are not reused. |
| Professor | Linear, MLP, bilinear/tensor, and late fusion | Partial | Common head implementations accepting either scratch or CLIP embeddings; shape/parameter tests. |
| Professor | Unimodal baselines and oracle | Partial | Image-only, text-only, and matched multimodal baselines plus explicitly named legacy/text-aware oracles. |
| Professor | Sample-efficiency curves | Partial/invalid protocol | Nested, stratified training subsets with fixed validation/test samples per seed. |
| Professor | Noise or missing-modality test | Partial | Matched corruption conditions and a consistent missingness policy for every model. |
| Professor | Cost reporting | Partial | Total/trainable parameters, extraction time, training time, peak memory, hardware/software metadata. |
| Professor | At least three seeds, mean +/- standard deviation | Missing | Raw per-seed rows plus aggregate tables; no claim from a single seed. |
| Professor | Zero-shot CLIP where labels are meaningfully textual | Missing | Scenario applicability table, fixed prompts, no privileged variables, recorded prompt set. |
| Professor | CPU/free-tier feasibility and about 5,000 samples/scenario | Partial | Smoke/full configs, chunked extraction, measured feasibility report. |
| Professor | One command/notebook regenerates figures and tables | Missing | A single documented reproduction command from configs and saved raw results. |
| Rubric | Fair comparisons and no test tuning | Missing | Shared sample IDs/splits, shared heads, validation-only selection, locked test evaluation. |
| Track 1 context | Distinguish representation from fusion failure | Architectural support only | Frozen-embedding probes and matched-head comparisons are supported; the full H1-H3 experiments remain Phase 2. |

## 4. Scientific and technical review of TP1

### 4.1 Reproducibility risks

1. **Model initialization is not seeded correctly.** `fit()` calls `torch.manual_seed(seed)` after a `FusionNet` has already been constructed. The seed controls batch permutations and modality masks, but not that model's initial weights.
2. **The experiment seed is not propagated consistently.** `torch_suite(..., seed=...)` changes the dataset but calls `fit()` without passing that seed. Scikit-learn MLP and GBM use `random_state=0` for every dataset seed, while logistic regression does not declare a random state.
3. **A single global RNG stream couples operations.** The generator uses one RNG sequentially for latent variables and every renderer. Adding text or changing image rendering would shift subsequent random draws and alter signal/tabular samples. Independent named RNG substreams are required.
4. **Generation and splitting reuse the same integer seed through separate generators.** This is deterministic, but it creates an unnecessary dependency between generated values and the split permutation. Use derived, named seeds for world, sample generation, split, rendering, initialization, data order, and corruption.
5. **No sample IDs or split manifests exist.** Cached embeddings cannot yet be proven to correspond to the same samples and ordering used by every model family.

### 4.2 Evaluation risks

1. **The sample-efficiency protocol changes the test set.** `sample_efficiency_C()` calls `load("C", n=n)` for each size, regenerating the complete dataset and test set. Those points are not evaluations on a common held-out population and the training sets are not nested.
2. **Most conclusions use one seed.** Several notebook interpretations use definitive language for small differences that may be random variation.
3. **Validation is not used for normal model selection.** PyTorch training uses fixed epochs with no validation checkpoint. Classical models use fixed settings. Conversely, late stacking trains its meta-learner on the validation set, leaving no independent validation data for its own selection.
4. **Late averaging is not calibrated.** Unimodal logit scales can differ, so equal logit averaging does not necessarily mean equal evidence. Calibration must be learned on validation data or this limitation must be explicit.
5. **Missing-modality conditions are asymmetric.** Late fusion is told which modality is absent and removes it; the joint model only receives zeros and no availability mask. Both should receive equivalent missingness information, or the conditions must be reported as different protocols.
6. **Test permutation importance is acceptable only as post-hoc analysis.** It must not drive hyperparameter selection, and uncertainty over seeds should be reported.
7. **Notebook results are not stored in a tidy result format.** Figures and tables cannot be regenerated independently of rerunning training.

### 4.3 Representation and oracle risks

1. **The current oracle is not a strict observable-data Bayes ceiling.** It reads the noisy latent observations before pixel/sample rendering noise. The source code already calls it slightly optimistic. It should be reported as `latent_observation_oracle`, not an unqualified oracle.
2. **Adding text changes the available information.** If text is generated from true latents, it may bypass noise and raise the ceiling. If it summarizes more than one informative latent, it can destroy XOR synergy by making text alone predictive.
3. **Upscaling alone may not make the synthetic images meaningful to CLIP.** A 224 px grayscale blob/grating is still far from CLIP's natural image distribution. A poor frozen result could reflect domain mismatch rather than loss of synergistic information. Phase 1 must document this limitation and include unimodal latent probes.
4. **Per-sample image normalization could erase informative amplitude.** Use a fixed global rendering transform, never normalization derived independently from each sample.
5. **Current handcrafted image features scale badly at 224 px.** Flattening produces 50,176 features per grayscale image. The legacy fixed-feature baselines should stay low-resolution; high-resolution direct comparisons should use learned encoders or controlled downsampling declared in the config.

### 4.4 Engineering risks

1. The current 224 px renderer constructs all images and several intermediates at once. Five thousand float32 grayscale images alone require roughly 1.0 GB; float32 RGB would require roughly 3.0 GB before temporary arrays.
2. Exact tensor fusion grows as `(projection_dim + 1) ** number_of_modalities`. It is manageable for two modalities but can become wasteful for four modalities.
3. The notebook assumes CPU tensors (`.numpy()` directly) and provides no device abstraction or batched data loader.
4. The notebook kernel records Python 3.14.7, but no compatible dependency versions are pinned.

## 5. Proposed benchmark extension

### 5.1 Separate benchmark variants

Use explicit variant names rather than silently changing A-E:

- `legacy_tp1`: exact current image-signal-tabular generator and low-resolution reference behavior.
- `vl_matched`: a derived image-text view designed for fair comparison across scratch, frozen CLIP, and adapted CLIP.
- `extended_four_modal` (optional Phase 1 compatibility check): image, signal, tabular, and text in one record. It should not be used to claim that CLIP beats a three-modality scratch model.

The required three-family comparison should use `vl_matched`, because every family can consume exactly the same image and text. The original benchmark remains available for regression and for demonstrating continuity with TP1.

### 5.2 Recommended `vl_matched` scenario semantics

| Scenario | Image channel | Text channel | Label | Scientific role |
|---|---|---|---|---|
| A - redundancy | Noisy observation of shared `s` | Independent noisy, quantized observation of the same `s` | `1[s > 0]` | Both modalities carry redundant evidence. |
| B - uniqueness | Noisy `u_img` | Noisy/quantized independent `u_txt` | `1[w_img*u_img + w_txt*u_txt > 0]` | Each modality contributes unique information. |
| C - synergy | Noisy bit `b_img` | Independently corrupted bit `b_txt` | `b_img XOR b_txt` | Main Phase 1 validation: each modality is marginally at chance. |
| D - imbalance | Clean `u_img` | Noisier `u_txt` | `1[u_img + u_txt > 0]` | Tests domination by the easier modality. |
| E - corruption | Same as A | Same as A | `1[s > 0]` | Train clean; corrupt/drop either modality at test time. |

Signal and tabular channels may still be generated for the optional four-modal record, but they are excluded from the matched image-text comparison. Each result row must record the benchmark variant and modality set.

### 5.3 Text generation contract

1. Introduce a separate text-channel observation with its own named RNG and noise parameter.
2. For Scenario C, generate a corrupted observed bit using a declared flip probability. The caption states only that bit. It never mentions the image bit, XOR relation, class, or target.
3. For continuous scenarios, quantize only the assigned text observation into balanced semantic categories such as left/right or low/high. Store the pre-quantized observation and the emitted category.
4. Select templates and synonyms independently of target conditional on the stated category. Counterbalance template IDs across categories and splits.
5. Avoid shortcuts from sentence length, punctuation, template frequency, vocabulary size, or ordered template IDs. Where practical, use paired templates with matching token counts.
6. Persist `text_observation`, `text_category`, `template_id`, and `text` alongside the original true latents and observed latents.
7. Add automated audits: class balance, per-modality predictability, duplicate captions, template-label association, caption-length association, and a bag-of-words shortcut baseline.

Generating captions from noisy observations is preferred because it makes text a real measurement channel. Generating from true latents is simpler, but it silently gives text privileged denoised information and changes the oracle more drastically.

### 5.4 Image rendering contract

- Preserve the Gaussian-blob position and oriented-grating nuisance relationships.
- Make resolution explicit in config and support at least legacy 16 px plus foundation-compatible 224 px.
- Render in deterministic chunks using independent per-sample/per-modality seeds, so chunk size and modality order cannot change content.
- Apply a fixed global clip-and-scale transform to uint8. Replicate grayscale to three RGB channels at load time unless an approved pilot establishes a scientifically justified color attribute.
- Use the official CLIP preprocessing after the benchmark's fixed rendering transform. Record checkpoint, processor revision, resize/crop, normalization, and code/config hashes in cache metadata.
- Prefer on-demand rendering or uint8 grayscale storage. Do not hold the full float32 RGB dataset in memory.
- Add invariance tests: the same latent/sample ID renders identically across runs and chunk sizes; changing resolution preserves normalized blob position and grating orientation.

### 5.5 Oracle policy

- Keep the legacy value for regression but label it `latent_observation_oracle`.
- Define a separate text-aware oracle for each extended scenario using the information actually assigned to the image/text channels.
- For Scenario C with independent symmetric bit corruption, compute the Bayes decision from the two channel observations; the oracle must degrade as either flip/noise rate grows.
- For quantized text, do not let the oracle read an unavailable continuous text value unless it is explicitly labeled an optimistic latent ceiling.
- Add invariant tests: unimodal Scenario C accuracy is approximately 0.5, joint oracle exceeds chance, and captions alone do not predict the XOR label.

## 6. Model architecture for the three mandatory families

### 6.1 Shared interfaces

All families should expose modality embeddings and use the same fusion-head interface:

`batch -> modality encoders -> optional common-dimension projections -> fusion -> binary head`

Shared heads:

- image-only and text-only linear/MLP classifiers;
- linear concatenation;
- concatenation plus a one-hidden-layer MLP;
- projected exact bilinear fusion plus a linear head;
- decision-level late fusion with validation calibration.

The data split, modality set, fusion projection dimension, head architecture, training budget, and selection rule must be explicit in every result row.

### 6.2 Family 1 - from scratch

- Adapt the existing convolutional image encoder to 224 px using strided blocks and adaptive pooling; do not flatten pixels.
- Add a deliberately small text encoder: trainable token embeddings, masked mean pooling, and an MLP projection. A small GRU is an optional ablation, not a Phase 1 requirement.
- Reuse/refactor the existing fusion modules only after legacy regression tests exist.
- Train unimodal and multimodal versions under the same validation and seed protocol.

### 6.3 Family 2 - frozen foundation encoders

- Initial checkpoint: CLIP ViT-B/32, one pinned implementation/revision.
- Freeze both encoders, run batched inference once, and cache float32 (or verified float16) image/text embeddings by stable sample ID.
- Cache keys must include dataset variant/version, scenario, data seed, sample manifest hash, renderer/preprocessor version, checkpoint/revision, modality, and embedding dtype.
- Normalize embeddings only where required by the defined method. Save both the policy and whether normalization occurred.
- Train exactly the same family of projection/fusion heads used for the scratch embeddings.
- Before fusion claims, run linear/nonlinear probes for the individual assigned latent attribute. These are Phase 1 diagnostics; the complete representation study remains Phase 2.

### 6.4 Family 3 - lightweight adaptation

Recommended first method: unfreeze the final transformer block and projection in the image and text encoders, keeping all earlier blocks frozen.

Why this choice:

- It meets the requirement that encoder parameters actually change.
- It avoids adding a PEFT/LoRA dependency before the basic pipeline is stable.
- Frozen prefixes reduce activation memory relative to full fine-tuning.
- It is easy to audit by listing trainable parameter names and comparing pre/post weights.

Consequences:

- Cached final frozen embeddings cannot be used.
- The pipeline must read images and tokenized text in small batches.
- Total and trainable parameters, peak memory, and training time must be reported separately.
- LoRA remains a Phase 2 alternative if final-block tuning is too costly or if a tighter parameter budget is required.

### 6.5 Bilinear feasibility

For two modalities, projection dimension 16 gives 289 tensor terms and dimension 32 gives 1,089 terms, both inexpensive. For four modalities, dimension 16 gives 83,521 terms per sample. Therefore exact bilinear fusion should use the matched two-modality configuration in Phase 1; four-modal tensor fusion requires a small projection or a documented low-rank alternative.

## 7. Fair comparison and unified evaluation protocol

### 7.1 Seed and split policy

- Use at least experiment seeds `0, 1, 2` for the full report.
- Derive named sub-seeds using `SeedSequence`: latent generation, split, each renderer, template selection, model initialization, data order, corruption, and subset selection.
- Generate a stable sample ID for every record and persist train/validation/test manifests.
- Use one fixed validation and test set per experiment seed.
- Construct stratified nested subsets from the training split only, for example 5%, 10%, 25%, 50%, and 100%. A larger subset must contain every sample in smaller subsets.

### 7.2 Training and selection

- Select hyperparameters and checkpoints only on validation data.
- Define one small common search space before test evaluation. Record every tried configuration.
- Use the same stopping rule and maximum budget for comparable heads.
- For late fusion, train/calibrate component models without consuming the only validation set for meta-training; use out-of-fold training predictions or a designated meta-training portion of the training split.
- Lock the test set until the selected configuration is fixed. Produce a separate `test` command or explicit lock flag.

### 7.3 Required outputs

For every applicable scenario and model family:

- accuracy per seed, mean, and standard deviation;
- optional balanced accuracy and negative log-likelihood as diagnostics;
- image-only, text-only, and multimodal results;
- text-aware/latent-observation oracle with an exact definition;
- sample-efficiency curve on fixed test samples;
- one predeclared corruption or missing-modality curve;
- total and trainable parameter counts;
- embedding extraction time, head/adaptation training time, inference time, peak memory, device, and software versions.

Raw results should be append-only tidy rows (CSV or Parquet) plus configuration snapshots. Tables and figures must read those rows; plotting code must not retrain models.

### 7.4 Zero-shot applicability

- Apply a CLIP zero-shot baseline only when the target is directly expressible from the available image through non-privileged class prompts, for example a left-versus-right image attribute.
- Predeclare prompt templates and average prompt embeddings if ensembling is used.
- Do not call an image-only CLIP prompt baseline valid for Scenario C when the target depends on an independent text bit.
- Do not put hidden latents, the ground-truth class, or the XOR operands into an inference prompt.
- Record `not_applicable` with a reason rather than forcing a misleading zero-shot number.

### 7.5 Phase 1 experiment matrix

Minimum full matrix:

1. Exact legacy TP1 regression on A-C with stored-output tolerance checks; E corruption as one robustness reference.
2. `vl_matched` A, B, C, D, E dataset/oracle validation.
3. Three model families on the main `vl_matched` C task with identical image-text inputs and common heads.
4. A smaller sanity comparison on A (redundancy) and E (corruption) to show the pipeline is not XOR-specific.
5. Three-seed, nested sample-efficiency and one corruption/missingness curve for the selected Phase 1 configurations.

Running every head for every family on every scenario is optional only after the minimum matrix is stable and measured. It must not delay a correct, reproducible core.

## 8. Proposed repository structure after approval

```text
.
|-- README.md
|-- pyproject.toml
|-- configs/
|   `-- phase1/
|       |-- smoke.yaml
|       `-- full.yaml
|-- docs/
|   |-- planning/
|   |   `-- phase-1-tp1-extension-plan.md
|   |-- references/
|   |   |-- professor-project-specification.pdf
|   |   `-- track-1-research-proposal.pdf
|   `-- decisions.md
|-- legacy/
|   `-- tp1/
|       |-- README.md
|       |-- multimodal_gen.py
|       `-- multimodal_fusion_lab_student.ipynb
|-- src/
|   `-- synergy_bench/
|       |-- data/
|       |   |-- generator.py
|       |   |-- render.py
|       |   |-- text.py
|       |   `-- splits.py
|       |-- models/
|       |   |-- scratch.py
|       |   |-- foundation.py
|       |   |-- adaptation.py
|       |   `-- fusion.py
|       |-- evaluation/
|       |   |-- metrics.py
|       |   |-- costs.py
|       |   `-- runner.py
|       `-- cli.py
|-- tests/
|   |-- test_legacy_regression.py
|   |-- test_dataset_semantics.py
|   |-- test_reproducibility.py
|   |-- test_fusion_shapes.py
|   `-- test_cache_integrity.py
|-- artifacts/                 # ignored; generated locally
|   |-- manifests/
|   |-- embeddings/
|   |-- checkpoints/
|   |-- results/
|   |-- figures/
|   `-- tables/
`-- scripts/
    `-- reproduce_phase1.sh
```

Migration rules:

- Move the notebook and generator together so their relative import remains valid; do not edit their contents during the move.
- Rename reference PDFs descriptively without changing them.
- Tag or commit the current root state before migration.
- Do not commit model weights, generated images, embedding caches, or full results unless a deliberate release policy is approved.
- Keep modules small and experiment-oriented. Avoid a framework abstraction that makes it hard to inspect the scientific path from latents to result.

No migration in this section should happen until this plan is approved.

## 9. Step-by-step implementation roadmap

### Milestone 0 - Freeze and document the TP1 baseline

Dependencies: plan approval.  
Work: tag current state, record file hashes, move legacy files together, add environment manifest, extract stored TP1 reference metrics.  
Acceptance criteria:

- Legacy notebook opens and imports its colocated generator.
- Source files and notebook outputs are byte-identical to the approved baseline.
- A documented environment can import NumPy, pandas, scikit-learn, matplotlib, and PyTorch.

### Milestone 1 - Reproducibility foundation

Dependencies: Milestone 0.  
Work: configs, named seed derivation, stable sample IDs, persisted split manifests, tidy result schema, smoke CLI.  
Acceptance criteria:

- Two identical smoke commands produce identical manifests, initial weights, metrics, and result rows.
- Changing only the model seed leaves data/splits unchanged.
- Changing renderer chunk size leaves samples unchanged.

### Milestone 2 - Legacy regression pipeline

Dependencies: Milestone 1.  
Work: wrap or minimally port reusable dataset/model/evaluation pieces; do not rewrite the notebook first.  
Acceptance criteria:

- A-C reference metrics match stored notebook outputs within a predeclared tolerance under the legacy single-seed config.
- Known methodological differences are documented rather than hidden.
- The sample-efficiency implementation demonstrably uses fixed validation/test manifests and nested training subsets.

### Milestone 3 - Extended image-text dataset

Dependencies: Milestone 1; approval of text/oracle decisions.  
Work: `vl_matched` A-E, text templates, 224 px chunked renderer, text-aware oracle, leakage audits.  
Acceptance criteria:

- Scenario unit tests pass for determinism, class balance, channel roles, and oracle behavior.
- In C, image-only and text-only latent-channel predictors are approximately chance, while the joint oracle is above chance.
- Template ID and caption length do not predict the label beyond a predeclared tolerance.
- Peak memory stays within the approved smoke/full budget.

### Milestone 4 - From-scratch image-text baselines

Dependencies: Milestone 3.  
Work: scratch encoders, shared heads, validation training, parameter/cost logging.  
Acceptance criteria:

- Unimodal baselines run for all A-E smoke configs.
- Nonlinear/bilinear fusion can solve an easy low-noise C smoke case; linear fusion remains near chance as expected.
- Every run reports total/trainable parameters and timing.

### Milestone 5 - Frozen CLIP extraction and heads

Dependencies: Milestones 3-4; checkpoint approval.  
Work: pinned CLIP integration, image/text embedding cache, cache validation, common heads, zero-shot applicability.  
Acceptance criteria:

- A repeated extraction request hits the cache and produces identical embeddings/sample order.
- Cache metadata mismatch causes a hard failure, not silent reuse.
- Frozen encoders have zero trainable parameters; heads train without encoder forward passes.
- Latent probes reveal whether each required operand is recoverable before interpreting fusion failure.

### Milestone 6 - Lightweight adaptation family

Dependencies: Milestone 5.  
Work: final-block/projection tuning, small-batch loader, frozen-versus-adapted weight audit, shared nonlinear head.  
Acceptance criteria:

- Only approved encoder parameters and the head require gradients.
- At least one encoder weight changes; frozen-prefix weights remain identical.
- No cached final embedding is used during adaptation.
- A smoke run completes within the approved memory/time budget.

### Milestone 7 - Unified Phase 1 evaluation

Dependencies: Milestones 4-6.  
Work: three seeds, nested subsets, corruption/missingness, aggregation, figures/tables from raw rows.  
Acceptance criteria:

- All direct comparisons use identical sample IDs, modalities, and splits.
- Test evaluation occurs only for validation-selected configs.
- Aggregates reproduce exactly from raw result rows without retraining.
- Mean +/- standard deviation and cost fields are present for every required result.

### Milestone 8 - Reproduction and handoff

Dependencies: Milestone 7.  
Work: final README, one-command script, clean-clone smoke/full instructions, limitations and AI-use declaration.  
Acceptance criteria:

- One documented command regenerates every Phase 1 figure and table from either cached artifacts or a full run.
- A clean smoke run passes tests and produces the expected artifact tree.
- Phase 2 hooks are documented, but no H1-H3 conclusion is claimed from Phase 1 alone.

## 10. Computational feasibility

### 10.1 Storage and memory estimates

- 5,000 x 224 x 224 float32 grayscale images: about 1.0 GB.
- The same images as float32 RGB: about 3.0 GB.
- The same images as uint8 grayscale: about 251 MB.
- Two 512-dimensional float32 CLIP embeddings for 5,000 samples: about 20.5 MB per scenario/seed before metadata.
- Five scenarios x three seeds of those two embedding sets: about 307 MB.

Recommendation: persist latents, captions, manifests, and compact uint8/on-demand images; cache embeddings. Do not persist full float32 RGB tensors.

### 10.2 Expected bottlenecks and controls

| Bottleneck | Risk | Control |
|---|---|---|
| High-resolution rendering | Large temporary arrays | Chunked deterministic rendering; uint8 output; on-demand loading. |
| CLIP extraction | Dominant initial compute | One checkpoint, batched inference, mixed precision where verified, immutable cache. |
| Adaptation | GPU memory and runtime | Final-block tuning, small batches, gradient accumulation, early smoke benchmarks. |
| Exact tensor fusion | Exponential feature growth | Two-modality use; small projection; parameter/activation checks before launch. |
| Multi-seed matrix | Accidental experiment explosion | Smoke/full configs and minimum matrix before optional expansion. |
| Repeated hyperparameter runs | Test leakage and cost | Predeclared compact grid, validation-only selection, saved run registry. |

The first compute checkpoint should measure 100-sample rendering/extraction and one adaptation epoch, then extrapolate. If a full configuration exceeds free-tier limits, reduce optional scenario/head breadth before reducing the required three seeds or corrupting comparison fairness.

## 11. Decisions requiring team approval

The following defaults are recommended; implementation should pause at the relevant milestone if the team rejects one.

1. **Benchmark design:** approve separate `legacy_tp1` and `vl_matched` variants rather than injecting text into the old scenarios under the same names.
2. **Main Phase 1 task:** approve image-bit XOR text-bit as the matched Scenario C, with signal/tabular excluded from direct three-family comparisons.
3. **Text information source:** approve captions from noisy text observations rather than true latents.
4. **Visual design:** approve preserving grayscale blob/grating semantics with fixed RGB channel replication for the first implementation. A color attribute should be added only if a predeclared pilot shows CLIP cannot meaningfully process the original visual semantics.
5. **Oracle terminology:** approve reporting the old value as `latent_observation_oracle` and adding a separately defined text-aware oracle.
6. **Foundation checkpoint:** approve one pinned CLIP ViT-B/32 implementation for Phase 1.
7. **Adaptation method:** approve final transformer block plus projection tuning; defer LoRA unless feasibility measurements justify it.
8. **Scenario breadth:** approve the minimum full matrix in Section 7.5 instead of every model/head/scenario combination.
9. **Artifact policy:** decide whether embedding caches and raw result tables are local-only, attached to a release, or stored with Git LFS/external storage.
10. **Missing TP1 report:** provide the standalone completed report if it exists, so its claims can be cross-checked against code and notebook outputs before implementation.

## 12. Phase boundary

Phase 1 ends when the benchmark extension, three model families, fair evaluation protocol, caches, cost logging, and reproduction command all work on the approved minimum matrix. Phase 1 may produce diagnostic observations, but it must not present the proposal's H1-H3 as tested conclusions.

Phase 2 will use this foundation for the complete synergy experiments, representation probes, fine-grained versus simple visual attributes, architectural ablations, and hypothesis-level statistical interpretation. Those tasks are intentionally not implemented by this plan.
