#!/usr/bin/env python3
"""
Script to run integration-based center vs edge classification.

This script uses the integration classifier to distinguish between traces from
center and edge positions based on integrated channel signals.

Data structure expected:
    base_data_path/
    ├── center/
    │   ├── ER_traces_energy_{energy}_pair_batch_XXXX.h5
    │   ├── NR_traces_energy_{energy}_pair_batch_XXXX.h5
    │   └── ...
    └── edge/
        ├── ER_traces_energy_{energy}_pair_batch_XXXX.h5
        ├── NR_traces_energy_{energy}_pair_batch_XXXX.h5
        └── ...
"""

from pathlib import Path
import sys
import argparse

# Add parent directory to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from reconstruction_model.integration_classifier import (
    IntegrationConfig,
    run_center_edge_classification,
    run_classification_pipeline
)


def main():
    """
    Run integration-based classification between center and edge positions.
    
    Uses trapezoidal integration (proper calculus integration) and XGBoost
    with BCE loss (binary:logistic objective).
    
    Center positions are labeled as 0, edge positions as 1.
    """
    parser = argparse.ArgumentParser(
        description="Center vs Edge classification using integrated channel signals"
    )
    parser.add_argument(
        "--energy", type=int, default=500,
        help="Energy level to select (e.g., 500, 5000). Default: 500"
    )
    parser.add_argument(
        "--recoil-types", type=str, nargs="+", default=["ER", "NR"],
        help="Recoil types to include (e.g., ER NR). Default: ER NR"
    )
    parser.add_argument(
        "--n-traces", type=int, default=1000,
        help="Maximum number of traces per class. Default: 1000"
    )
    parser.add_argument(
        "--base-data-path", type=str, 
        default="/ceph/dwong/trigger_samples/sanity_check_Jan12",
        help="Base path containing center and edge folders"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results. Default: {PROJECT_ROOT}/integration_results"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of boosting rounds (like epochs). Default: 100"
    )
    parser.add_argument(
        "--max-depth", type=int, default=6,
        help="Maximum tree depth. Default: 6"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1,
        help="Learning rate. Default: 0.1"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = _PROJECT_ROOT / "integration_results"
    else:
        output_dir = Path(args.output_dir)
    
    # Configuration
    config = IntegrationConfig(
        n_traces=args.n_traces,
        n_channels=56,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        
        # Data paths
        base_data_path=args.base_data_path,
        center_folder="center",
        edge_folder="edge",
        
        # Energy and recoil type selection
        energy=args.energy,
        recoil_types=args.recoil_types,
        
        # XGBoost hyperparameters
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        early_stopping_rounds=10,
    )
    
    print("=" * 80)
    print("Integration-Based Center vs Edge Classification")
    print("=" * 80)
    print(f"Base data path: {config.base_data_path}")
    print(f"Energy: {config.energy}")
    print(f"Recoil types: {config.recoil_types}")
    print(f"Max traces per class: {config.n_traces}")
    print(f"Integration method: Trapezoidal (numerical calculus)")
    print(f"Loss function: BCE (binary:logistic)")
    print(f"Boosting rounds: {config.n_estimators}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Verify data paths exist
    base_path = Path(config.base_data_path)
    center_path = base_path / config.center_folder
    edge_path = base_path / config.edge_folder
    
    if not base_path.exists():
        print(f"ERROR: Base data path not found: {base_path}")
        return
    
    if not center_path.exists():
        print(f"ERROR: Center folder not found: {center_path}")
        return
    
    if not edge_path.exists():
        print(f"ERROR: Edge folder not found: {edge_path}")
        return
    
    # Run the center vs edge classification pipeline
    metrics = run_center_edge_classification(
        config=config,
        output_dir=str(output_dir)
    )
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1']:.2%}")
    print("=" * 80)
    print("\nConfusion Matrix:")
    print(f"  [[TN, FP], [FN, TP]]")
    print(f"  {metrics['confusion_matrix']}")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
