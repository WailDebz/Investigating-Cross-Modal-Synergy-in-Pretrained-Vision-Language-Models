# Decisions (Phase 1.1: data, reproducibility and oracles)

Status values: **decided** (team decision) or **to confirm** (most conservative option, applied provisionally; to be reviewed by the team / Student B).

## D1 - Legacy file hashes and line endings (decided)

`legacy/tp1/SHA256SUMS.txt` and the command in `legacy/tp1/README.md` hash the content stored by Git (blob content, LF line endings). On Windows checkouts with `core.autocrlf=true` the working-tree files have CRLF line endings, so a raw file hash differs. The regression tests (`tests/test_legacy_regression.py`) and the adapter (`synergy_bench.data.legacy.normalized_sha256`) therefore hash the file after converting CRLF to LF. Both methods give the same values:

| File | SHA-256 |
|---|---|
| `multimodal_gen.py` | `e222d35a191a3bed3c29ae1fb7f5b51e310f0055d81a97c4296462bc7d4ebac3` |
| `multimodal_fusion_lab_student.ipynb` | `b0d5f9560e45090bea20d5934ea57148c6116504c31b9be9e8644624bb8f19e1` |

## D2 - Legacy adapter (decided)

- `synergy_bench.data.legacy` loads `legacy/tp1/multimodal_gen.py` by path after checking its hash. The source is executed in memory, so no bytecode is written into `legacy/`. The generator is never copied or edited.
- `legacy_tp1` manifests use the contract sample IDs and the new stratified splits from `split_seed`, not the legacy `split_dataset` permutation.
- The legacy oracle is reported as `latent_observation_oracle`.
- Stored notebook outputs (single seed) are kept in `NOTEBOOK_REFERENCE_METRICS`. Data-level values (oracles, class balance) depend only on the generator and are tested with tolerance `5e-4` (print rounding). All of them reproduce.
- **To confirm:** model-level notebook metrics (sklearn grid, torch suite, noise sweep) are stored for the Phase 1.2 end-to-end regression with tolerance `0.03` (single-seed historical values).
- Rendered legacy arrays (`img`, `sig`, `tab`) are checked by float64 sums with relative tolerance `1e-6`, not by exact hash, because `exp`/`cos` may differ in the last bit across CPUs. RNG-only arrays (`y`, `obs`, latents, oracle predictions) are checked by exact SHA-256.

## D3 - Protocol differences between the TP1 notebook and the new pipeline

| Aspect | TP1 notebook | New pipeline |
|---|---|---|
| Random streams | one RNG consumed in sequence by all steps | named, independent streams (`seeds.py`), per sample for latents, rendering and templates |
| Generation vs split seed | same integer seed for both | `data_seed` and `split_seed` are separate |
| Split | random permutation, `int(f * n)` cuts, not stratified | stratified by target, fixed by `split_seed`, persisted manifest |
| Sample identity | row position | stable `sample_id` |
| Sample efficiency | regenerates the whole dataset (new test set) per size | nested, class-balanced subsets of the fixed training split |
| Model init seed | `torch.manual_seed` after model construction | Phase 1.2 (`model_init` stream) |
| Seeds | single seed | at least three experiment seeds (Phase 1.2) |
| Oracle name | `oracle_acc` | `latent_observation_oracle` (legacy) and `text_aware_oracle` (vl_matched) |
| Modalities | image, signal, tabular | `vl_matched`: image and text only |
| Image | 16 px float32, grayscale | `render_v1` at configurable size (224 px), fixed uint8 mapping, RGB by replication |
| Missing modality | late fusion is told what is missing; joint model gets zeros | availability flags in every item; one policy for every model (Phase 1.2) |

## D4 - `vl_matched` latent model (decided, with one point to confirm)

The text channel replaces the legacy signal channel; signal and tabular are not generated. `obs_img = z_img + sigma_img * e`, `obs_txt = z_txt + sigma_txt * e'`, `sigma_txt = legacy signal sigma * text.noise`, `sigma_img = legacy image sigma`.

| Scenario | Latents | Target | (sigma_img, sigma_txt base) |
|---|---|---|---|
| A, E | `z_img = z_txt = s ~ N(0,1)` | `1[s > 0]` | (0.8, 0.8) |
| B | `z_img, z_txt ~ N(0,1)` | `1[z_img + z_txt > 0]` | (0.5, 0.5) |
| C | `z = 2b - 1`, `b ~ Bernoulli(1/2)` | `b_img XOR b_txt` | (0.6, 0.6) |
| D | as B | `1[z_img + z_txt > 0]` | (0.2, 0.6) |

- **To confirm:** B uses the legacy weights of the two kept channels (1, 1); the tabular weight 0.7 disappears with the tabular channel.
- **To confirm:** each sample draws a fixed set of values from `make_rng(data_seed, "latents", index)`. A sample's content therefore does not depend on `n_samples` or on the scenario (A and E share latents, captions and images, as in the legacy generator; only their sample IDs differ).
- The image renders `obs_img` (blob position) and an independent nuisance (grating orientation), with legacy pixel noise `0.1 + 0.2 * sigma_img`.

## D5 - Text channel and templates (decided)

- The caption states only `text_category = 1[obs_txt > 0]`, computed from the text channel's own noisy observation. For C this flips `b_txt` with probability `Phi(-1 / sigma_txt)` (0.048 at `text.noise = 1`).
- Template set `v1`: four frames shared by both categories, filled with the neutral code word `maple` (category 0) or `cedar` (category 1). No position, orientation, colour or shape word. Both words have 5 characters and 1 whitespace token, so for a given frame both categories give captions of identical length.
- `template_id` is drawn uniformly from `make_rng(data_seed, "text_template", index)`, independently of the category and the target.
- **To confirm:** "token" means whitespace token; the BPE token counts of `maple`/`cedar` in the CLIP tokenizer have not been checked yet (Phase 1.2).
- Captions come from a closed set of 8 strings, so caption duplication is by design; the audits check for duplicate sample IDs and duplicate latent rows instead.

## D6 - Metadata access (decided)

`load_split(bundle, split, with_metadata=False)`: without the flag `item["metadata"] == {}`; with it, the analysis metadata (true and noisy latents, nuisance, text category, template ID, oracle prediction, split, render version, template set). The same values are always available in the separate latent manifest (`bundle.latents`, `latents.csv` in the cache folder), never in the contract manifest.

## D7 - Images (decided, with one point to confirm)

- Grayscale, replicated to RGB at load time; no colour attribute.
- **To confirm with Student B:** `item["image"] = uint8 / 255` as float32 in [0, 1], `[C, H, W]`, without mean/std normalization. Model-specific preprocessing (e.g. CLIP resize/normalize) is applied by Phase 1.2 on top of this fixed transform.
- **To confirm:** `image.channels` accepts 1 (legacy-style grayscale) or 3 (contract default).

## D8 - uint8 mapping, storage and memory (decided, with one point to confirm)

- `render_v1`: clip to the fixed global range `[-1.5, 2.5]`, then scale to 0..255 and round. The range covers blob plus grating plus more than 4.5 standard deviations of the largest pixel noise (0.26); measured clipping is below 0.1% of pixels.
- Rendering is on demand per sample (`make_rng(data_seed, "image_render", index)`, phase drawn first, then pixel noise). An optional uint8 grayscale memmap cache is written under a caller-given folder (to be placed in `artifacts/`), keyed by the config hash and render version, and written atomically. The cache folder also receives the manifest, latent manifest, canonical config and metadata.
- **To confirm:** declared peak-memory budget for rendering at 224 px: 64 MB, independent of `n_samples` (tested with `tracemalloc`). Storage of the 5,000-sample cache: about 251 MB of uint8 on disk.

## D9 - Dependencies and audits (decided, with one point to confirm)

- No dependency added. Audits (`synergy_bench.data.audits`) use NumPy only: class balance, unimodal latent predictability, oracle accuracy, template frequencies, caption lengths, duplicates, and shortcut diagnostics fitted on train and scored on test (majority per `template_id`, majority per token count, bag-of-words logistic regression).
- **To confirm:** chance tolerance `|accuracy - 0.5| <= 0.05` (about three standard errors on a 1,000-sample test split).

## D10 - Text-aware oracle (decided, with one point to confirm)

- `text_aware_oracle` is the Bayes decision for `y` given `obs_img` and the caption category under the declared generative model. For C it is closed form (`sign(obs_img) XOR category`, expected accuracy `a_i a_t + (1 - a_i)(1 - a_t)`, `a = Phi(1 / sigma)`, 0.909 at default noise). For A, B, D and E it is a threshold on `obs_img` that depends on the category, found by bisection on a 1-D quadrature of the posterior.
- It never reads the continuous `obs_txt`. Its accuracy decreases monotonically when either channel's noise increases (tested).
- **To confirm:** like the legacy oracle, it reads the latent image observation before pixel noise, so it is an optimistic ceiling for image models. This limitation is written into the dataset metadata (`metadata["oracle"]["limitations"]`).

## D11 - Integration fixture (decided, with one point to confirm)

`tests/fixtures/data_contract/`: `images.npz` (compressed uint8 grayscale images and sample IDs) and `contract.json` (contract description, configs, config and manifest hashes, split counts, oracle info, and per item the sample ID, split, target, caption, availability and analysis metadata). It covers scenarios A, C and E of the reference configs (5,000 samples, 224 px), with the first two samples of each split. Total size about 0.8 MB. Regenerate with `python -m synergy_bench.data.fixture tests/fixtures/data_contract`. `load_fixture_items` rebuilds contract items from the fixture alone.

- **To confirm:** regenerated images must match the fixture within one uint8 level (cross-CPU differences in `exp`/`cos`); everything else must match exactly.

## D12 - Scope boundaries (to confirm)

- `build_dataset` supports `vl_matched` only; `legacy_tp1` is served by `synergy_bench.data.legacy` (and `load_manifest` supports both variants).
- Scenario E items are clean (availability all true). Test-time corruption and missingness are applied by the Phase 1.2 evaluation with the `corruption` stream of `data_seed`. Proposed missing-modality representation: image of zeros, empty caption, and the corresponding availability flag set to false.
