"""
Training script for XGBoost Position + Energy Regressor.

This script trains an XGBoost model to predict event positions (x, y, z) AND energy
from integrated channel signals. The integration reduces the full trace
to a single value per channel, creating a compact feature vector.

Usage:
    python -m reconstruction_model.train_xgboost
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from reconstruction_model.dataset import (
    DataConfig,
    ParticleReconstructionDataset,
    create_dataloaders,
)
from reconstruction_model.model import (
    XGBoostPositionRegressor, 
    XGBoostRegressorConfig,
    XGBoostPositionEnergyRegressor,
    XGBoostPositionEnergyRegressorConfig,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_BASE_CHECKPOINT_PATH = _PROJECT_ROOT / "training_checkpoints"
_BASE_CHECKPOINT_PATH.mkdir(exist_ok=True, parents=True)


@dataclass 
class XGBoostTrainingConfig:
    """Configuration for XGBoost training run"""
    # Data settings
    target_energy: Optional[Union[int, List[int]]] = None  # Filter by energy: int, list, or None for all
    recoil_types: list = field(default_factory=lambda: ["ER", "NR"])
    access_mode: str = "remote"  # "local" or "remote"
    local_data_path: str | Path = _PROJECT_ROOT / "data"
    remote_data_path: str = "/ceph/srv/ssjostrom/training_small"
    
    # Model mode
    predict_position: bool = True  # If True, predict position (x, y, z)
    predict_energy: bool = True    # If True, predict energy
    
    # Normalization settings
    normalise_positions: bool = False
    normalise_energy: bool = False  # Energy normalization (z-score)
    precomputed_positions: bool = False
    precomputed_energy: bool = False
    
    # Integration settings  
    integration_dt: float = 2.56e-7
    
    # XGBoost hyperparameters for position
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05

    # Energy-specific hyperparameters
    energy_n_estimators: int = 500
    energy_max_depth: int = 6
    energy_learning_rate: float = 0.01

    # Shared hyperparameters
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    
    
    # Training settings
    random_state: int = 42
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    
    # Output settings
    checkpoint_dir: str | Path = _BASE_CHECKPOINT_PATH
    save_plots: bool = True
    plot_dir: str | Path = _PROJECT_ROOT / "plots"
    
    def __post_init__(self):
        if not self.predict_position and not self.predict_energy:
            raise ValueError("At least one of predict_position or predict_energy must be True")
        
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.plot_dir = Path(self.plot_dir)
        self.plot_dir.mkdir(exist_ok=True, parents=True)


def prepare_integrated_data(
    dataset: ParticleReconstructionDataset,
    max_samples: Optional[int] = None,
    integration_dt: float = 2.56e-7,
    verbose: bool = True,
    include_energy: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load traces from dataset and integrate them to create feature vectors.
    
    Args:
        dataset: ParticleReconstructionDataset instance
        max_samples: Maximum number of samples to load (None = all)
        integration_dt: Time step for numerical integration
        verbose: Whether to show progress bar
        include_energy: Whether to return energy targets and sample energies
    
    Returns:
        Tuple of (X, y_pos, y_energy, sample_energies) where:
            X: Integrated features of shape (n_samples, n_channels)
            y_pos: Position targets of shape (n_samples, 3)
            y_energy: Energy targets of shape (n_samples,) - normalized if applicable
            sample_energies: Original (unnormalized) energy values of shape (n_samples,)
    """
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    X_list = []
    y_pos_list = []
    y_energy_list = []
    sample_energy_list = []  # Original energy values for per-energy analysis
    
    iterator = tqdm(range(n_samples), desc="Integrating traces") if verbose else range(n_samples)
    
    for idx in iterator:
        trace, spatial_target, energy_target, recoil_type = dataset[idx]
        
        # trace is (n_channels, trace_length) tensor
        trace_np = trace.numpy()
        
        # Integrate using trapezoidal rule
        integrated = XGBoostPositionEnergyRegressor.integrate_traces(trace_np, dt=integration_dt)
        
        X_list.append(integrated)
        y_pos_list.append(spatial_target.numpy())
        
        if include_energy:
            y_energy_list.append(energy_target.numpy())
            # Get original energy from sample_info for per-energy analysis
            sample_info = dataset.split_samples[idx]
            sample_energy_list.append(sample_info['energy'])
    
    X = np.array(X_list)
    y_pos = np.array(y_pos_list)
    y_energy = np.array(y_energy_list) if include_energy else None
    sample_energies = np.array(sample_energy_list) if include_energy else None
    
    if verbose:
        logger.info(f"Prepared {n_samples} samples")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y_pos shape: {y_pos.shape}")
        if include_energy:
            logger.info(f"  y_energy shape: {y_energy.shape}")
            unique_energies = np.unique(sample_energies)
            logger.info(f"  Unique energies: {sorted(unique_energies)}")
    
    return X, y_pos, y_energy, sample_energies


def plot_training_curves(
    training_history: dict,
    save_path: Optional[Path] = None,
    show: bool = False,
    include_energy: bool = True
) -> None:
    """
    Plot training and validation RMSE curves for each coordinate and energy.
    
    Args:
        training_history: Dictionary with training history
        save_path: Path to save the plot
        show: Whether to display the plot
        include_energy: Whether to include energy curve
    """
    n_plots = 4 if include_energy else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    coords = ['x', 'y', 'z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for ax, coord, color in zip(axes[:3], coords, colors):
        train_key = f'train_rmse_{coord}'
        val_key = f'val_rmse_{coord}'
        
        train_rmse = training_history.get(train_key, [])
        val_rmse = training_history.get(val_key, [])
        
        if train_rmse:
            iterations = range(len(train_rmse))
            ax.plot(iterations, train_rmse, label='Train', color=color, alpha=0.8)
        
        if val_rmse:
            iterations = range(len(val_rmse))
            ax.plot(iterations, val_rmse, label='Validation', color=color, 
                   linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Boosting Iteration')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{coord.upper()} Coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Energy curve
    if include_energy and len(axes) > 3:
        ax = axes[3]
        train_rmse = training_history.get('train_rmse_energy', [])
        val_rmse = training_history.get('val_rmse_energy', [])
        
        if train_rmse:
            ax.plot(range(len(train_rmse)), train_rmse, label='Train', color=colors[3], alpha=0.8)
        if val_rmse:
            ax.plot(range(len(val_rmse)), val_rmse, label='Validation', color=colors[3], linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Boosting Iteration')
        ax.set_ylabel('RMSE')
        ax.set_title('Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('XGBoost Training Curves', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = False,
    title_suffix: str = ""
) -> None:
    """
    Create scatter plots comparing true vs predicted positions.
    
    Args:
        y_true: True positions of shape (n_samples, 3)
        y_pred: Predicted positions of shape (n_samples, 3)
        save_path: Path to save the plot
        show: Whether to display the plot
        title_suffix: Additional text for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    coords = ['X', 'Y', 'Z']
    units = ['mm', 'mm', 'mm']
    
    for i, (ax, coord, unit) in enumerate(zip(axes, coords, units)):
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        # Scatter plot
        ax.scatter(true_vals, pred_vals, alpha=0.3, s=10, c='#1f77b4')
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect', linewidth=2)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
        mae = np.mean(np.abs(pred_vals - true_vals))
        
        ax.set_xlabel(f'True {coord} [{unit}]')
        ax.set_ylabel(f'Predicted {coord} [{unit}]')
        ax.set_title(f'{coord}: RMSE={rmse:.2f}, MAE={mae:.2f} {unit}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')
        
        # Remove decimal points from Z-axis tick labels
        if i == 2:  # Z coordinate
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
    
    plt.suptitle(f'XGBoost Position Prediction - 500eV {title_suffix}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Prediction scatter saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_error_distribution(
    spatial_errors: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = False,
    title_suffix: str = ""
) -> None:
    """
    Plot distribution of 3D spatial errors.
    
    Args:
        spatial_errors: Array of 3D Euclidean errors
        save_path: Path to save the plot
        show: Whether to display the plot
        title_suffix: Additional text for title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(spatial_errors, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(spatial_errors), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(spatial_errors):.2f} mm')
    axes[0].axvline(np.median(spatial_errors), color='green', linestyle='--',
                    label=f'Median: {np.median(spatial_errors):.2f} mm')
    axes[0].set_xlabel('3D Spatial Error [mm]')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_errors = np.sort(spatial_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    axes[1].plot(sorted_errors, cumulative, color='#1f77b4', linewidth=2)
    
    # Add percentile markers
    percentiles = [50, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(spatial_errors, p)
        axes[1].axhline(p, color='gray', linestyle=':', alpha=0.5)
        axes[1].axvline(val, color='gray', linestyle=':', alpha=0.5)
        axes[1].scatter([val], [p], color='red', s=50, zorder=5)
        axes[1].annotate(f'{p}%: {val:.1f}mm', (val, p), 
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    axes[1].set_xlabel('3D Spatial Error [mm]')
    axes[1].set_ylabel('Cumulative Percentage')
    axes[1].set_title('Cumulative Error Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Spatial Error Analysis {title_suffix}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error distribution saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(
    model,
    save_path: Optional[Path] = None,
    show: bool = False,
    top_k: int = 20,
    include_energy: bool = True
) -> None:
    """
    Plot feature importance for each coordinate and energy model.
    
    Args:
        model: Trained XGBoostPositionRegressor or XGBoostPositionEnergyRegressor
        save_path: Path to save the plot
        show: Whether to display the plot
        top_k: Number of top features to show
        include_energy: Whether to include energy importance
    """
    importance = model.get_feature_importance()
    
    has_energy = 'energy' in importance and importance['energy'] is not None
    n_plots = 4 if (include_energy and has_energy) else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6))
    
    coords = ['x', 'y', 'z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for ax, coord, color in zip(axes[:3], coords, colors):
        imp = importance[coord]
        
        # Get top-k features
        top_indices = np.argsort(imp)[-top_k:][::-1]
        top_values = imp[top_indices]
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(top_indices))
        ax.barh(y_pos, top_values, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Ch {i}' for i in top_indices])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{coord.upper()} Coordinate (Top {top_k})')
        ax.grid(True, alpha=0.3, axis='x')
    
    # Energy importance
    if include_energy and has_energy and len(axes) > 3:
        ax = axes[3]
        imp = importance['energy']
        top_indices = np.argsort(imp)[-top_k:][::-1]
        top_values = imp[top_indices]
        
        y_pos = np.arange(len(top_indices))
        ax.barh(y_pos, top_values, color=colors[3], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Ch {i}' for i in top_indices])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Energy (Top {top_k})')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Channel Feature Importance', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_energy_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_energies: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    title_suffix: str = ""
) -> None:
    """
    Create comprehensive energy prediction plots.
    
    Args:
        y_true: True energy values
        y_pred: Predicted energy values
        sample_energies: Original (unnormalized) energy values for grouping
        save_path: Path to save the plot
        show: Whether to display the plot
        title_suffix: Additional text for title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter plot: True vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, c='#d62728')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect', linewidth=2)
    
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))
    ax.set_xlabel('True Energy')
    ax.set_ylabel('Predicted Energy')
    ax.set_title(f'Energy Prediction: RMSE={rmse:.4f}, MAE={mae:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Error histogram
    ax = axes[0, 1]
    errors = y_pred - y_true
    ax.hist(errors, bins=50, color='#d62728', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='-', linewidth=2)
    ax.axvline(np.mean(errors), color='blue', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
    ax.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.4f}')
    ax.set_xlabel('Prediction Error (Pred - True)')
    ax.set_ylabel('Count')
    ax.set_title('Energy Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Residuals vs True
    ax = axes[1, 0]
    ax.scatter(y_true, errors, alpha=0.3, s=10, c='#d62728')
    ax.axhline(0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('True Energy')
    ax.set_ylabel('Residual (Pred - True)')
    ax.set_title('Residuals vs True Energy')
    ax.grid(True, alpha=0.3)
    
    # 4. Per-energy boxplot or MAPE by energy
    ax = axes[1, 1]
    if sample_energies is not None:
        unique_energies = np.unique(sample_energies)
        energy_errors = []
        labels = []
        
        for energy in sorted(unique_energies):
            mask = sample_energies == energy
            if np.sum(mask) > 0:
                energy_errors.append(np.abs(errors[mask]))
                labels.append(f'{int(energy)}')
        
        bp = ax.boxplot(energy_errors, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#d62728')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('True Energy')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error Distribution by Energy Level')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        # If no sample energies, show percentiles
        abs_errors = np.abs(errors)
        percentiles = [25, 50, 75, 90, 95, 99]
        values = [np.percentile(abs_errors, p) for p in percentiles]
        ax.bar(range(len(percentiles)), values, color='#d62728', alpha=0.7)
        ax.set_xticks(range(len(percentiles)))
        ax.set_xticklabels([f'P{p}' for p in percentiles])
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error Percentiles')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Energy Prediction Analysis {title_suffix}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Energy prediction plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_per_energy_metrics(
    metrics: dict,
    save_path: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Plot performance metrics broken down by energy level.
    
    Args:
        metrics: Dictionary containing 'per_energy' key with per-energy metrics
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if 'per_energy' not in metrics:
        logger.warning("No per_energy metrics available")
        return
    
    per_energy = metrics['per_energy']
    energies = sorted(per_energy.keys())
    
    if not energies:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    n_samples = [per_energy[e]['n_samples'] for e in energies]
    pos_mean_errors = [per_energy[e]['pos_mean_error'] for e in energies]
    pos_median_errors = [per_energy[e]['pos_median_error'] for e in energies]
    energy_rmse = [per_energy[e]['energy_rmse'] for e in energies]
    energy_mape = [per_energy[e]['energy_mape'] for e in energies]
    energy_bias = [per_energy[e]['energy_bias'] for e in energies]
    
    x_pos = np.arange(len(energies))
    width = 0.35
    
    # 1. Sample distribution
    ax = axes[0, 0]
    bars = ax.bar(x_pos, n_samples, color='#1f77b4', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(e) for e in energies])
    ax.set_xlabel('Energy Level')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Distribution by Energy')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Position errors
    ax = axes[0, 1]
    ax.bar(x_pos - width/2, pos_mean_errors, width, label='Mean', color='#ff7f0e', alpha=0.8)
    ax.bar(x_pos + width/2, pos_median_errors, width, label='Median', color='#2ca02c', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(e) for e in energies])
    ax.set_xlabel('Energy Level')
    ax.set_ylabel('3D Position Error')
    ax.set_title('Position Error by Energy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight struggling energies (above mean)
    mean_pos_error = np.mean(pos_mean_errors)
    for i, (x, err) in enumerate(zip(x_pos, pos_mean_errors)):
        if err > mean_pos_error * 1.2:  # 20% above mean
            ax.annotate('!', (x - width/2, err), ha='center', fontsize=14, color='red', fontweight='bold')
    
    # 3. Energy RMSE
    ax = axes[1, 0]
    colors_bar = ['#d62728' if r > np.mean(energy_rmse) * 1.2 else '#2ca02c' for r in energy_rmse]
    ax.bar(x_pos, energy_rmse, color=colors_bar, alpha=0.8)
    ax.axhline(np.mean(energy_rmse), color='black', linestyle='--', label=f'Mean: {np.mean(energy_rmse):.4f}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(e) for e in energies])
    ax.set_xlabel('Energy Level')
    ax.set_ylabel('Energy RMSE')
    ax.set_title('Energy Prediction RMSE by Energy Level')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Energy MAPE and Bias
    ax = axes[1, 1]
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x_pos - width/2, energy_mape, width, label='MAPE (%)', color='#9467bd', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, energy_bias, width, label='Bias', color='#8c564b', alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(e) for e in energies])
    ax.set_xlabel('Energy Level')
    ax.set_ylabel('MAPE (%)', color='#9467bd')
    ax2.set_ylabel('Bias', color='#8c564b')
    ax.set_title('Energy MAPE and Bias by Energy Level')
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Performance Metrics by Energy Level\n(Red bars indicate energies struggling > 20% above mean)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Per-energy metrics plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def train_xgboost_regressor(config: XGBoostTrainingConfig):
    """
    Main training function for XGBoost position + energy regressor.
    
    Args:
        config: Training configuration
    
    Returns:
        Trained XGBoostPositionEnergyRegressor or XGBoostPositionRegressor model
    """
    # Determine model type
    if config.predict_position and config.predict_energy:
        model_type = "Position + Energy"
    elif config.predict_position:
        model_type = "Position Only"
    else:
        model_type = "Energy Only"
    
    logger.info("="*70)
    logger.info(f"XGBoost {model_type} Regressor Training")
    logger.info("="*70)
    
    # Log target energy configuration
    if config.target_energy is None:
        logger.info("  Target Energy: All energies")
    elif isinstance(config.target_energy, list):
        logger.info(f"  Target Energies: {config.target_energy}")
    else:
        logger.info(f"  Target Energy: {config.target_energy}")
    
    # Set random seed
    np.random.seed(config.random_state)
    
    # Create data configuration
    data_config = DataConfig(
        access_mode=config.access_mode,
        local_data_path=config.local_data_path,
        remote_data_path=config.remote_data_path,
        recoil_types=config.recoil_types,
        target_energy=config.target_energy,
        normalise_inputs=False,  # We don't need trace normalization for integration
        normalise_positions=config.normalise_positions if config.predict_position else False,
        normalise_energy=config.normalise_energy if config.predict_energy else False,
    )
    
    logger.info("\nCreating datasets...")
    dataloaders = create_dataloaders(
        data_config=data_config,
        batch_size=32,
        num_workers=0,
        precomputed_trace=False,
        precomputed_positions=config.precomputed_positions if config.predict_position else False,
        precomputed_energy=config.precomputed_energy if config.predict_energy else False,
    )
    
    train_dataset = dataloaders['train'].dataset
    val_dataset = dataloaders['val'].dataset
    test_dataset = dataloaders['test'].dataset
    
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # Store normalisers for later use
    target_normaliser = train_dataset.target_normaliser
    energy_mean = train_dataset.energy_mean
    energy_std = train_dataset.energy_std
    
    # Log normalization status
    if config.predict_position:
        if target_normaliser is not None:
            logger.info("\n✓ Position normalisation enabled")
            logger.info(f"  Stats: X_mean={target_normaliser.means[0]:.2f}, Y_mean={target_normaliser.means[1]:.2f}, Z_mean={target_normaliser.means[2]:.2f}")
            logger.info(f"         X_std={target_normaliser.stds[0]:.2f}, Y_std={target_normaliser.stds[1]:.2f}, Z_std={target_normaliser.stds[2]:.2f}")
        else:
            logger.info("\n✗ Position normalisation disabled")
    
    if config.predict_energy and energy_mean is not None:
        logger.info(f"✓ Energy normalisation enabled (mean={energy_mean:.2f}, std={energy_std:.2f})")
    elif config.predict_energy:
        logger.info("✗ Energy normalisation disabled")
    
    # Prepare integrated data
    logger.info("\nPreparing training data (integrating traces)...")
    start_time = time.time()
    
    X_train, y_train_pos, y_train_energy, train_sample_energies = prepare_integrated_data(
        train_dataset, 
        max_samples=config.max_train_samples,
        integration_dt=config.integration_dt,
        include_energy=True  # Always load energy for sample tracking
    )
    
    X_val, y_val_pos, y_val_energy, val_sample_energies = prepare_integrated_data(
        val_dataset,
        max_samples=config.max_val_samples,
        integration_dt=config.integration_dt,
        include_energy=True
    )
    
    X_test, y_test_pos, y_test_energy, test_sample_energies = prepare_integrated_data(
        test_dataset,
        max_samples=config.max_test_samples,
        integration_dt=config.integration_dt,
        include_energy=config.predict_energy
    )
    
    prep_time = time.time() - start_time
    logger.info(f"Data preparation completed in {prep_time:.2f}s")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if config.predict_position and config.predict_energy:
        # ============================
        # POSITION + ENERGY MODEL
        # ============================
        xgb_config = XGBoostPositionEnergyRegressorConfig(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            energy_n_estimators=config.energy_n_estimators,
            energy_max_depth=config.energy_max_depth,
            energy_learning_rate=config.energy_learning_rate,
            early_stopping_rounds=config.early_stopping_rounds,
            random_state=config.random_state,
            integration_dt=config.integration_dt,
            model_save_dir=config.checkpoint_dir,
        )
        
        model = XGBoostPositionEnergyRegressor(xgb_config)
        model.target_normaliser = target_normaliser
        model.energy_mean = energy_mean
        model.energy_std = energy_std
        
        logger.info("\nTraining XGBoost Position + Energy models...")
        start_time = time.time()
        
        training_history = model.fit(
            X_train, y_train_pos, y_train_energy,
            X_val=X_val, y_val_pos=y_val_pos, y_val_energy=y_val_energy,
            verbose=True
        )
        
        train_time = time.time() - start_time
        logger.info(f"\nTraining completed in {train_time:.2f}s")
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        metrics = model.evaluate(
            X_test, y_test_pos, y_test_energy,
            sample_energies=test_sample_energies,
            verbose=True
        )
        
        # Save model
        model_save_path = config.checkpoint_dir / f"xgboost_pos_energy_{timestamp}"
        model.save(model_save_path)
        
        # Generate plots
        if config.save_plots:
            logger.info("\nGenerating plots...")
            
            # Training curves (with energy)
            plot_training_curves(
                training_history,
                save_path=config.plot_dir / f"xgboost_training_curves_{timestamp}.png",
                include_energy=True
            )
            
            # Position prediction scatter
            plot_prediction_scatter(
                y_test_pos, metrics['predictions_pos'],
                save_path=config.plot_dir / f"xgboost_position_scatter_{timestamp}.png",
                title_suffix=f"(Test Set, n={len(y_test_pos)})"
            )
            
            # Position error distribution
            plot_error_distribution(
                metrics['spatial_errors'],
                save_path=config.plot_dir / f"xgboost_position_errors_{timestamp}.png",
                title_suffix=f"(Test Set)"
            )
            
            # Energy prediction analysis
            plot_energy_prediction(
                y_test_energy, metrics['predictions_energy'],
                sample_energies=test_sample_energies,
                save_path=config.plot_dir / f"xgboost_energy_prediction_{timestamp}.png",
                title_suffix=f"(Test Set, n={len(y_test_energy)})"
            )
            
            # Per-energy breakdown
            plot_per_energy_metrics(
                metrics,
                save_path=config.plot_dir / f"xgboost_per_energy_metrics_{timestamp}.png"
            )
            
            # Feature importance (with energy)
            plot_feature_importance(
                model,
                save_path=config.plot_dir / f"xgboost_feature_importance_{timestamp}.png",
                include_energy=True
            )
    
    elif config.predict_position:
        # ============================
        # POSITION ONLY MODEL
        # ============================
        xgb_config = XGBoostRegressorConfig(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            early_stopping_rounds=config.early_stopping_rounds,
            random_state=config.random_state,
            integration_dt=config.integration_dt,
            model_save_dir=config.checkpoint_dir,
        )
        
        model = XGBoostPositionRegressor(xgb_config)
        model.target_normaliser = target_normaliser
        
        logger.info("\nTraining XGBoost Position models...")
        start_time = time.time()
        
        training_history = model.fit(
            X_train, y_train_pos,
            X_val=X_val, y_val=y_val_pos,
            verbose=True
        )
        
        train_time = time.time() - start_time
        logger.info(f"\nTraining completed in {train_time:.2f}s")
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test_pos, verbose=True)
        
        # Save model
        model_save_path = config.checkpoint_dir / f"xgboost_position_{timestamp}"
        model.save(model_save_path)
        
        # Generate plots
        if config.save_plots:
            logger.info("\nGenerating plots...")
            
            plot_training_curves(
                training_history,
                save_path=config.plot_dir / f"xgboost_training_curves_{timestamp}.png",
                include_energy=False
            )
            
            plot_prediction_scatter(
                y_test_pos, metrics['predictions'],
                save_path=config.plot_dir / f"xgboost_position_scatter_{timestamp}.png",
                title_suffix=f"(Test Set, n={len(y_test_pos)})"
            )
            
            plot_error_distribution(
                metrics['spatial_errors'],
                save_path=config.plot_dir / f"xgboost_position_errors_{timestamp}.png",
                title_suffix=f"(Test Set)"
            )
            
            plot_feature_importance(
                model,
                save_path=config.plot_dir / f"xgboost_feature_importance_{timestamp}.png",
                include_energy=False
            )
    
    else:  # Energy only
        # ============================
        # ENERGY ONLY MODEL
        # ============================
        # Use a single XGBoost regressor for energy
        import xgboost as xgb
        
        logger.info("\nTraining XGBoost Energy model...")
        start_time = time.time()
        
        # Create energy model
        model = xgb.XGBRegressor(
            n_estimators=config.energy_n_estimators,
            max_depth=config.energy_max_depth,
            learning_rate=config.energy_learning_rate,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=1,
        )
        
        # Train
        eval_set = [(X_train, y_train_energy)]
        if X_val is not None and y_val_energy is not None:
            eval_set.append((X_val, y_val_energy))
        
        model.fit(X_train, y_train_energy, eval_set=eval_set, verbose=True)
        
        train_time = time.time() - start_time
        logger.info(f"\nTraining completed in {train_time:.2f}s")
        
        # Predict and evaluate
        logger.info("\nEvaluating on test set...")
        y_pred_energy = model.predict(X_test)
        
        # Compute metrics
        energy_rmse = np.sqrt(np.mean((y_pred_energy - y_test_energy) ** 2))
        energy_mae = np.mean(np.abs(y_pred_energy - y_test_energy))
        energy_mape = np.mean(np.abs((y_pred_energy - y_test_energy) / (np.abs(y_test_energy) + 1e-8))) * 100
        
        ss_res = np.sum((y_test_energy - y_pred_energy) ** 2)
        ss_tot = np.sum((y_test_energy - np.mean(y_test_energy)) ** 2)
        energy_r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        logger.info("\n" + "="*70)
        logger.info("Energy Regressor - Evaluation Results")
        logger.info("="*70)
        logger.info(f"  RMSE:  {energy_rmse:.4f}")
        logger.info(f"  MAE:   {energy_mae:.4f}")
        logger.info(f"  MAPE:  {energy_mape:.2f}%")
        logger.info(f"  R²:    {energy_r2:.4f}")
        
        # Per-energy breakdown
        if test_sample_energies is not None:
            unique_energies = np.unique(test_sample_energies)
            logger.info("\n  PER-ENERGY BREAKDOWN:")
            logger.info("  " + "-"*60)
            logger.info(f"  {'Energy':>8} | {'N':>5} | {'RMSE':>8} | {'MAE':>8} | {'MAPE':>7}")
            logger.info("  " + "-"*60)
            for energy in sorted(unique_energies):
                mask = test_sample_energies == energy
                if np.sum(mask) > 0:
                    e_rmse = np.sqrt(np.mean((y_pred_energy[mask] - y_test_energy[mask]) ** 2))
                    e_mae = np.mean(np.abs(y_pred_energy[mask] - y_test_energy[mask]))
                    e_mape = np.mean(np.abs((y_pred_energy[mask] - y_test_energy[mask]) / (np.abs(y_test_energy[mask]) + 1e-8))) * 100
                    logger.info(f"  {int(energy):>8} | {int(np.sum(mask)):>5} | {e_rmse:>8.4f} | {e_mae:>8.4f} | {e_mape:>6.2f}%")
            logger.info("  " + "-"*60)
        logger.info("="*70)
        
        # Save model
        model_save_path = config.checkpoint_dir / f"xgboost_energy_{timestamp}"
        model_save_path.mkdir(exist_ok=True, parents=True)
        model.save_model(str(model_save_path / "model_energy.json"))
        
        # Save energy normalizer if applicable
        if energy_mean is not None:
            import pickle
            energy_stats = {'mean': energy_mean, 'std': energy_std}
            with open(model_save_path / "energy_normaliser.pkl", 'wb') as f:
                pickle.dump(energy_stats, f)
        
        # Generate plots
        if config.save_plots:
            logger.info("\nGenerating plots...")
            
            # Create minimal metrics dict for plotting
            metrics = {
                'predictions_energy': y_pred_energy,
            }
            
            # Energy prediction analysis
            plot_energy_prediction(
                y_test_energy, y_pred_energy,
                sample_energies=test_sample_energies,
                save_path=config.plot_dir / f"xgboost_energy_prediction_{timestamp}.png",
                title_suffix=f"(Test Set, n={len(y_test_energy)})"
            )
            
            # Feature importance
            importance = model.feature_importances_
            fig, ax = plt.subplots(1, 1, figsize=(8, 10))
            top_k = 20
            top_indices = np.argsort(importance)[-top_k:][::-1]
            top_values = importance[top_indices]
            
            y_pos = np.arange(len(top_indices))
            ax.barh(y_pos, top_values, color='#d62728', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f'Ch {i}' for i in top_indices])
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Energy Model - Channel Importance (Top {top_k})')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(config.plot_dir / f"xgboost_energy_feature_importance_{timestamp}.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Feature importance saved")
    
    logger.info("\n" + "="*70)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {model_save_path}")
    if config.save_plots:
        logger.info(f"Plots saved to: {config.plot_dir}")
    logger.info("="*70)
    
    return model


def main():
    """Main entry point for XGBoost training"""
    # Create training configuration
    config = XGBoostTrainingConfig(
        # Data settings
        access_mode="remote",
        target_energy=[500],  # List of energies, single int, or None for all
        recoil_types=["NR"],
        
        # Model mode (at least one must be True)
        predict_position=True,  
        predict_energy=False,    
        
        # Normalization
        normalise_positions=False,
        normalise_energy=False,  # Set to True for z-score normalization of energy
        precomputed_positions=False,
        precomputed_energy=False,
        
        # XGBoost position hyperparameters
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        
        # XGBoost energy hyperparameters (can tune separately)
        energy_n_estimators=1000,
        energy_max_depth=10,
        energy_learning_rate=0.01,
        
        
        # Limits for quick testing (set to None for full dataset)
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        
        # Output
        save_plots=True,
    )
    
    # Train the model
    model = train_xgboost_regressor(config)
    
    return model


if __name__ == "__main__":
    main()
