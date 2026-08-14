# Legacy Reconstruction Utilities

This folder preserves useful scripts from the old `src/reconstruction_model/`
copy while keeping them separate from the active training path.

The active package entry points remain:

```text
reconstruction_model.train
reconstruction_model.dataset
reconstruction_model.models
```

Files in this folder may still reference old dataset names, checkpoint paths,
or optional dependencies. Treat them as migration/reference material until they
are adapted and tested against the current H5 training pipeline.

## Migrated Files

- `evaluation_metrics.py`: old evaluation helpers with RMSE/classification
  metric logic.
- `evaluate_checkpoint_legacy.py`: old checkpoint evaluation script.
- `visualise_legacy.py`: Plotly detector/prediction visualization utilities.
- `train_xgboost_legacy.py`: old XGBoost position/energy training script.
- `integration_classifier_legacy.py`: old integrated-trace XGBoost classifier.
- `normalisation_legacy.py`: old trace normalization helper.
- `compute_position_stats_legacy.py`: old position-statistics cache script.
- `run_integration_classification_legacy.py`: old CLI wrapper for the
  integrated-trace classifier.
