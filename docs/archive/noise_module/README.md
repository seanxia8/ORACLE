# Modular Noise Simulator

Install from the repository:

```bash
python -m pip install ./noise_module
```

The package provides stationary PSD synthesis, evolutionary noise, artifacts,
non-Gaussian models, target-CSD multichannel synthesis, confidence-aware
validation, empirical calibration presets, and deterministic streaming.

See `noise_module/CONFIGURATION.md` and `examples/end_to_end.py`.

Run the standalone verification suite and example from this directory:

```bash
pytest -q noise_module/tests
python -m examples.end_to_end
```

Version 0.3 uses strict configuration by default and requires an explicit
layout for two-dimensional calibration arrays. See `CHANGELOG.md` for the
schema-3 migration notes.
