# Legacy TP1 (frozen reference)

This folder contains the original TP1 material of the Multimodal Machine Learning module, preserved **unchanged** as the reference for the project.

| File | Role |
|---|---|
| `multimodal_gen.py` | Original synthetic generator: image / signal / tabular modalities, scenarios A-E, latent-observation oracle, `split_dataset`, `corrupt`, `handcrafted_features` |
| `multimodal_fusion_lab_student.ipynb` | Completed TP1 notebook with stored single-seed outputs |
| `SHA256SUMS.txt` | SHA-256 of both files, computed on the Git-stored content |

## Rules

- **Do not edit any file in this folder.** New research code lives in `src/synergy_bench/` and only wraps or reuses this generator.
- The two files must stay in the same folder: the notebook imports `multimodal_gen` with a relative import.
- The notebook outputs are historical single-seed observations. They are used as regression references, not as final research results.

## Reference values

Running `python multimodal_gen.py` from this folder must print these oracle accuracies (n = 5000, seed 0):

| Scenario | A | B | C | D | E |
|---|---|---|---|---|---|
| `oracle_acc` | 0.869 | 0.852 | 0.915 | 0.885 | 0.869 |

In the research pipeline this value is reported as `latent_observation_oracle`: it reads the noisy latent observations before pixel/sample rendering noise, so it is a slightly optimistic ceiling.

## Verify integrity

From the repository root:

```bash
python -c "import hashlib,subprocess;[print(hashlib.sha256(subprocess.check_output(['git','cat-file','blob','HEAD:legacy/tp1/'+f])).hexdigest()+'  '+f) for f in ['multimodal_gen.py','multimodal_fusion_lab_student.ipynb']]"
```

The output must match `SHA256SUMS.txt`. Hashes are computed on the Git blob content so that they do not depend on local line-ending conversion.

## Open the notebook

The notebook needs `numpy`, `pandas`, `matplotlib`, `torch` and `scikit-learn`. Start Jupyter from this folder so that the relative import resolves:

```bash
cd legacy/tp1
jupyter notebook multimodal_fusion_lab_student.ipynb
```

The notebook was originally run with Python 3.14.7; no dependency versions were pinned at the time.

## Known limitations (see `docs/planning/phase-1-tp1-extension-plan.md`, section 4)

- A single RNG stream is shared by all renderers, so adding a modality would shift every later random draw.
- Model initialization is not seeded inside `fit()`.
- The sample-efficiency sweep regenerates the full dataset, and therefore a new test set, for each training size.
- Most notebook conclusions rely on a single seed.

These issues are addressed in the new pipeline, not by editing these files.