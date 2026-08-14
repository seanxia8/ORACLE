# Empirical calibration workflow

1. Arrange input as `(records, channels, samples)` and create
   `ReferenceDataset(data, fs, dataset_id, units=..., acquisition=...,
   preprocessing=...)`.
2. Save the immutable input contract with `dataset.save("reference.npz")`.
3. Run `calibrate_dataset(dataset, train_fraction=0.75, nperseg=...,
   bootstrap_samples=...)`.
4. Inspect `preset.heldout_validation`; a failed preset must not be presented as
   validated without revising the predeclared claims or model.
5. Save with `preset.save("preset.json")`. The JSON includes source identity,
   acquisition/preprocessing, estimator settings, code revision, UTC date,
   uncertainty, claims, and explicitly unmodeled properties.
6. Recreate simulations with `preset.generate(N, seed=...)` or sample calibration
   uncertainty with `preset.sample_parameters(seed=...)`.

The calibration/held-out split is seeded and recorded. PSD uncertainty uses a
record bootstrap. Calibration and validation records never overlap.
