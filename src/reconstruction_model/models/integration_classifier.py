"""
Integration-based XGBoost classifier for trace position classification.

This module integrates traces over each channel and uses XGBoost to classify
between different positions based on the integrated signals.
"""

from __future__ import annotations
import logging
import glob
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

import numpy as np
import torch
import h5py
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for integration-based classification"""
    n_traces: int = 1000  # Number of traces per position (per class)
    n_channels: int = 56  # Number of channels
    test_size: float = 0.2
    val_size: float = 0.1  # Validation split from training data
    random_state: int = 42
    
    # Data paths
    base_data_path: str = "/ceph/dwong/trigger_samples/sanity_check_Jan12"
    center_folder: str = "center"  # Folder containing center position data (label=0)
    edge_folder: str = "edge"      # Folder containing edge position data (label=1)
    
    # Energy selection (e.g., 500, 5000)
    energy: int = 5000
    
    # Recoil type filter (None = all, "ER", "NR", or ["ER", "NR"])
    recoil_types: Optional[list] = None
    
    # XGBoost parameters (n_estimators is like epochs for boosting)
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 100  # Number of boosting rounds (trees)
    objective: str = "binary:logistic"  # BCE loss equivalent
    eval_metric: str = "logloss"  # Binary cross-entropy
    early_stopping_rounds: int = 10  # Stop if no improvement
    
    def __post_init__(self):
        if self.recoil_types is None:
            self.recoil_types = ["ER", "NR"]


class TraceIntegrator:
    """Integrates traces over channels"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
    
    def integrate_traces(self, traces: np.ndarray, dt: float = 2.56e-7) -> np.ndarray:
        """
        Integrate traces over each channel using trapezoidal rule (numerical integration).
        
        Args:
            traces: Array of shape (n_traces, n_channels, trace_length)
            dt: Time step between samples (default=1.0 for normalized time)
        
        Returns:
            Integrated values of shape (n_traces, n_channels)
        """
        # Trapezoidal integration - proper numerical integration from calculus
        integrated = np.trapezoid(traces, dx=dt, axis=2)
        return integrated
    
    def _validate_positions(self, positions: np.ndarray, position_label: str) -> dict:
        """
        Validate that all traces come from consistent positions.
        
        Args:
            positions: Array of shape (n_traces, 3) with x, y, z coordinates
            position_label: Label for logging (e.g., "Position 1")
        
        Returns:
            Dictionary with mean, std, and range statistics
        """
        mean_pos = positions.mean(axis=0)
        std_pos = positions.std(axis=0)
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        
        stats = {
            'mean': mean_pos,
            'std': std_pos,
            'min': min_pos,
            'max': max_pos,
            'range': max_pos - min_pos
        }
        
        logger.info(f"{position_label} statistics:")
        logger.info(f"  Mean (x, y, z): {mean_pos}")
        logger.info(f"  Std  (x, y, z): {std_pos}")
        logger.info(f"  Range (x, y, z): {stats['range']}")
        
        return stats
    
    def load_and_integrate_from_h5(
        self,
        file_path: str,
        max_traces: Optional[int] = None,
        position_label: str = "Position"
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load traces from H5 file and integrate them.
        
        Args:
            file_path: Path to H5 file
            max_traces: Maximum number of traces to load (None = all)
            position_label: Label for logging position statistics
        
        Returns:
            Tuple of (integrated_traces, positions, position_stats) where:
                - integrated_traces: shape (n_traces, n_channels)
                - positions: shape (n_traces, 3) - x, y, z coordinates
                - position_stats: dictionary with mean, std, range statistics
        """
        logger.info(f"Loading traces from {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            traces_dataset = f['traces']
            events_dataset = f['events']
            
            n_events = len(traces_dataset)
            if max_traces is not None:
                indices = np.arange(min(max_traces, n_events))
            else:
                indices = np.arange(n_events)
            
            # Load traces
            traces = traces_dataset[indices]  # (n_traces, n_channels, trace_length)
            
            # Load positions for validation and statistics
            positions = np.column_stack([
                events_dataset['x'][indices],
                events_dataset['y'][indices],
                events_dataset['z'][indices]
            ])
        
        logger.info(f"Loaded {len(traces)} traces with shape {traces.shape}")
        
        # Validate and log position statistics
        position_stats = self._validate_positions(positions, position_label)
        
        # Integrate using trapezoidal rule
        integrated = self.integrate_traces(traces)
        
        logger.info(f"Integrated to shape {integrated.shape}")
        
        return integrated, positions, position_stats
    
    def discover_h5_files(
        self,
        folder_path: str,
        energy: int,
        recoil_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Discover H5 files matching energy and recoil type criteria.
        
        Files follow pattern: {recoil_type}_traces_energy_{energy}_pair_batch_{batch}.h5
        
        Args:
            folder_path: Path to folder containing H5 files
            energy: Energy value to filter (e.g., 500, 5000)
            recoil_types: List of recoil types to include (e.g., ["ER", "NR"])
        
        Returns:
            List of matching file paths sorted by batch number
        """
        if recoil_types is None:
            recoil_types = ["ER", "NR"]
        
        matching_files = []
        folder = Path(folder_path)
        files_per_type = {}
        
        for recoil_type in recoil_types:
            pattern = f"{recoil_type}_traces_energy_{energy}_pair_batch_*.h5"
            files = sorted(folder.glob(pattern))
            files_per_type[recoil_type] = len(files)
            matching_files.extend([str(f) for f in files])
        
        # Log summary
        logger.info(f"Found {len(matching_files)} total files for energy={energy}")
        for recoil_type in recoil_types:
            count = files_per_type.get(recoil_type, 0)
            logger.info(f"  {recoil_type}: {count} files")
        
        # Show sample files
        if matching_files:
            logger.info("Sample files:")
            for f in matching_files[:3]:
                logger.info(f"  - {Path(f).name}")
            if len(matching_files) > 3:
                logger.info(f"  ... and {len(matching_files) - 3} more")
        else:
            logger.warning(f"⚠️ No files found matching pattern for energy={energy}, recoil_types={recoil_types}")
        
        return matching_files
    
    def load_and_integrate_from_multiple_h5(
        self,
        file_paths: List[str],
        max_traces: Optional[int] = None,
        position_label: str = "Position"
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load traces from multiple H5 files and integrate them.
        
        Args:
            file_paths: List of paths to H5 files
            max_traces: Maximum total number of traces to load (None = all)
            position_label: Label for logging position statistics
        
        Returns:
            Tuple of (integrated_traces, positions, position_stats) where:
                - integrated_traces: shape (n_traces, n_channels)
                - positions: shape (n_traces, 3) - x, y, z coordinates
                - position_stats: dictionary with mean, std, range statistics
        """
        all_traces = []
        all_positions = []
        total_loaded = 0
        
        # Count files by recoil type for logging
        recoil_counts = {'ER': 0, 'NR': 0}
        for fp in file_paths:
            fname = Path(fp).name
            if 'ER_traces' in fname:
                recoil_counts['ER'] += 1
            elif 'NR_traces' in fname:
                recoil_counts['NR'] += 1
        
        logger.info(f"Loading traces from {len(file_paths)} files for {position_label}")
        if recoil_counts['ER'] > 0:
            logger.info(f"  ER files: {recoil_counts['ER']}")
        if recoil_counts['NR'] > 0:
            logger.info(f"  NR files: {recoil_counts['NR']}")
        
        for file_path in tqdm(file_paths, desc=f"Loading {position_label}"):
            with h5py.File(file_path, 'r') as f:
                traces_dataset = f['traces']
                events_dataset = f['events']
                
                n_events = len(traces_dataset)
                logger.info(f"Used file: {Path(file_path).name}")

                # Check if we've reached max_traces limit
                if max_traces is not None:
                    remaining = max_traces - total_loaded
                    if remaining <= 0:
                        break
                    n_to_load = min(n_events, remaining)
                else:
                    n_to_load = n_events
                
                indices = np.arange(n_to_load)
                
                # Load traces
                traces = traces_dataset[indices]  # (n_traces, n_channels, trace_length)
                
                # Load positions
                positions = np.column_stack([
                    events_dataset['x'][indices],
                    events_dataset['y'][indices],
                    events_dataset['z'][indices]
                ])
                
                all_traces.append(traces)
                all_positions.append(positions)
                total_loaded += n_to_load
        
        # Concatenate all data
        all_traces = np.concatenate(all_traces, axis=0)
        all_positions = np.concatenate(all_positions, axis=0)
        
        logger.info(f"Loaded {len(all_traces)} traces with shape {all_traces.shape}")
        
        # Validate and log position statistics
        position_stats = self._validate_positions(all_positions, position_label)
        
        # Integrate using trapezoidal rule
        integrated = self.integrate_traces(all_traces)
        
        logger.info(f"Integrated to shape {integrated.shape}")
        logger.info(f"Sample integrated values (first trace, first 5 channels): {integrated[0, :5]}")
        return integrated, all_positions, position_stats


class PositionClassifier:
    """XGBoost classifier for position discrimination"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.model = None
        self.feature_importance = None
        self.evals_result = {}  # Store training history for loss curves
    
    def prepare_binary_dataset(
        self,
        features_pos1: np.ndarray,
        features_pos2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare binary classification dataset from two position groups.
        
        Args:
            features_pos1: Integrated features from position 1 (n_traces, n_channels)
            features_pos2: Integrated features from position 2 (n_traces, n_channels)
        
        Returns:
            Tuple of (X, y) where:
                - X: Combined features (2*n_traces, n_channels)
                - y: Labels (2*n_traces,) with 0 for pos1, 1 for pos2
        """
        logger.info(f"Position 1 features shape: {features_pos1.shape}")
        logger.info(f"Position 2 features shape: {features_pos2.shape}")
        
        pos1_channels = [0, 1, 2, 3, 4, 5, 6, 30, 31, 36, 37, 38, 43, 44]  # Channels near center
        
        pos2_channels = [21, 22, 26, 27, 5, 15, 16, 17]  # Channels near edge
        
        # Analyze all key channels and find the one with max discrimination
        logger.info("\n" + "="*70)
        logger.info("FINDING MOST DISCRIMINATIVE CHANNEL:")
        logger.info("="*70)
        
        max_ratio = 0
        best_channel = None
        
        
        # Check pos1-favoring channels (expect pos1 > pos2, so use pos1/pos2 ratio)
        for ch in pos1_channels:
            pos1_mean = features_pos1[:, ch].mean()
            pos2_mean = features_pos2[:, ch].mean()
            ratio = pos1_mean / (pos2_mean + 1e-10)
            
            if ratio > max_ratio:
                max_ratio = ratio
                best_channel = ch
                best_pos1_mean = pos1_mean
                best_pos2_mean = pos2_mean
        
        # Check pos2-favoring channels (expect pos2 > pos1, so use pos2/pos1 ratio)
        for ch in pos2_channels:
            pos1_mean = features_pos1[:, ch].mean()
            pos2_mean = features_pos2[:, ch].mean()
            ratio = pos2_mean / (pos1_mean + 1e-10)
            
            if ratio > max_ratio:
                max_ratio = ratio
                best_channel = ch
                best_pos1_mean = pos1_mean
                best_pos2_mean = pos2_mean
        
        # Log the winner
        logger.info(f"Most discriminative channel: Channel {best_channel}")
        logger.info(f"  Max ratio: {max_ratio:.2f}")
        logger.info(f"  Position 1 mean: {best_pos1_mean}")
        logger.info(f"  Position 2 mean: {best_pos2_mean}")
        

        X = np.vstack([features_pos1, features_pos2])
        y = np.hstack([
            np.zeros(len(features_pos1)),
            np.ones(len(features_pos2))
        ])
        
        return X, y
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """
        Train XGBoost classifier with validation monitoring.
        
        XGBoost uses binary:logistic objective which is equivalent to BCE loss.
        Training monitors both training and validation loss (logloss = BCE).
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features (required for monitoring)
            y_val: Validation labels (required for monitoring)
        """
        logger.info(f"Training XGBoost with {len(X_train)} samples, {X_train.shape[1]} features")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Loss function: {self.config.objective} (BCE loss)")
        logger.info(f"Eval metric: {self.config.eval_metric} (binary cross-entropy)")
        
        self.model = xgb.XGBClassifier(
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            objective=self.config.objective,  # binary:logistic = BCE loss
            eval_metric=self.config.eval_metric,  # logloss = BCE
            early_stopping_rounds=self.config.early_stopping_rounds,
            random_state=self.config.random_state,
            use_label_encoder=False
        )
        
        # Train with validation monitoring
        self.evals_result = {}  # Reset training history
        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True  # Shows validation loss each round
        )
        
        # Store training history from the model
        self.evals_result = self.model.evals_result()
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Log best iteration
        best_iter = self.model.best_iteration
        logger.info(f"Training complete. Best iteration: {best_iter}")
        logger.info(f"Stopped at {best_iter} out of {self.config.n_estimators} boosting rounds")
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: Optional[Path] = None
    ) -> dict:
        """
        Evaluate classifier on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Optional directory to save plots
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Compute precision-recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve
        }
        
        # Log results
        logger.info("=" * 60)
        logger.info("Classification Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  Avg Precision (PR AUC): {metrics['avg_precision']:.4f}")
        logger.info("=" * 60)
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{metrics['confusion_matrix']}")
        logger.info("=" * 60)
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Center', 'Edge'])}")
        logger.info("=" * 60)
        
        # Save plots if output directory provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Confusion matrix plot
            self._plot_confusion_matrix(metrics['confusion_matrix'], output_dir)
            
            # Feature importance plot
            if self.feature_importance is not None:
                self._plot_feature_importance(output_dir)
            
            # ROC curve
            self._plot_roc_curve(fpr, tpr, roc_auc, output_dir)
            
            # Precision-Recall curve
            self._plot_precision_recall_curve(precision_curve, recall_curve, avg_precision, output_dir)
            
            # Score distributions
            self._plot_score_distributions(y_test, y_pred_proba, output_dir)
            
            # Training/validation loss curves
            if self.evals_result:
                self._plot_training_curves(output_dir)
            
            # Combined diagnostic plot
            self._plot_diagnostic_summary(metrics, output_dir)
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray, output_dir: Path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Position 1', 'Position 2'],
            yticklabels=['Position 1', 'Position 2']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    
    def _plot_feature_importance(self, output_dir: Path, top_k: int = 20):
        """Plot and save feature importance"""
        if self.feature_importance is None:
            return
        
        # Get top k features
        indices = np.argsort(self.feature_importance)[-top_k:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_k), self.feature_importance[indices])
        plt.yticks(range(top_k), [f'Channel {i}' for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Most Important Channels')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300)
        plt.close()
        logger.info(f"Saved feature importance to {output_dir / 'feature_importance.png'}")
    
    def _plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_dir: Path):
        """Plot and save ROC curve with AUC"""
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve (Receiver Operating Characteristic)', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300)
        plt.close()
        logger.info(f"Saved ROC curve to {output_dir / 'roc_curve.png'}")
    
    def _plot_precision_recall_curve(
        self, 
        precision: np.ndarray, 
        recall: np.ndarray, 
        avg_precision: float, 
        output_dir: Path
    ):
        """Plot and save Precision-Recall curve"""
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='green', lw=2, 
                 label=f'PR curve (AP = {avg_precision:.4f})')
        plt.axhline(y=0.5, color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300)
        plt.close()
        logger.info(f"Saved Precision-Recall curve to {output_dir / 'precision_recall_curve.png'}")
    
    def _plot_score_distributions(
        self, 
        y_test: np.ndarray, 
        y_pred_proba: np.ndarray, 
        output_dir: Path
    ):
        """Plot classifier score distributions for each class - key for detecting overfitting"""
        center_scores = y_pred_proba[y_test == 0]
        edge_scores = y_pred_proba[y_test == 1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram plot
        ax1 = axes[0]
        ax1.hist(center_scores, bins=30, alpha=0.6, color='blue', label='Center (label=0)', 
                 density=True, edgecolor='darkblue')
        ax1.hist(edge_scores, bins=30, alpha=0.6, color='red', label='Edge (label=1)', 
                 density=True, edgecolor='darkred')
        ax1.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Decision boundary')
        ax1.set_xlabel('Predicted Probability (Edge)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Classifier Score Distributions', fontsize=14)
        ax1.legend(loc='upper center', fontsize=10)
        ax1.set_xlim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Statistics
        center_mean, center_std = center_scores.mean(), center_scores.std()
        edge_mean, edge_std = edge_scores.mean(), edge_scores.std()
        
        # Check for potential overfitting
        separation = abs(edge_mean - center_mean)
        overlap_indicator = (center_std + edge_std) / 2
        
        stats_text = (f"Center: μ={center_mean:.3f}, σ={center_std:.3f}\n"
                      f"Edge: μ={edge_mean:.3f}, σ={edge_std:.3f}\n"
                      f"Separation: {separation:.3f}")
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # KDE plot for smoother visualization
        ax2 = axes[1]
        if len(np.unique(center_scores)) > 1:
            sns.kdeplot(center_scores, ax=ax2, color='blue', fill=True, alpha=0.4, label='Center')
        else:
            ax2.axvline(x=center_scores[0], color='blue', lw=3, label=f'Center (all={center_scores[0]:.3f})')
        
        if len(np.unique(edge_scores)) > 1:
            sns.kdeplot(edge_scores, ax=ax2, color='red', fill=True, alpha=0.4, label='Edge')
        else:
            ax2.axvline(x=edge_scores[0], color='red', lw=3, label=f'Edge (all={edge_scores[0]:.3f})')
        
        ax2.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Decision boundary')
        ax2.set_xlabel('Predicted Probability (Edge)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Score Distributions (KDE)', fontsize=14)
        ax2.legend(loc='upper center', fontsize=10)
        ax2.set_xlim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'score_distributions.png', dpi=300)
        plt.close()
        logger.info(f"Saved score distributions to {output_dir / 'score_distributions.png'}")
        
        # Log warning if scores are too extreme (potential overfitting indicator)
        if center_std < 0.01 or edge_std < 0.01:
            logger.warning("⚠️ Very low score variance detected - model may be overconfident!")
            logger.warning(f"  Center score std: {center_std:.4f}, Edge score std: {edge_std:.4f}")
        
        if (center_scores.max() < 0.1 and edge_scores.min() > 0.9):
            logger.warning("⚠️ Perfect separation detected - check for data leakage or overfitting!")
    
    def _plot_training_curves(self, output_dir: Path):
        """Plot training and validation loss curves"""
        if not self.evals_result:
            logger.warning("No training history available for loss curves")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Get metrics from eval results
        metric_name = self.config.eval_metric
        
        # Training set is validation_0, validation set is validation_1
        train_losses = self.evals_result.get('validation_0', {}).get(metric_name, [])
        val_losses = self.evals_result.get('validation_1', {}).get(metric_name, [])
        
        if not train_losses or not val_losses:
            logger.warning(f"Could not find {metric_name} in training history")
            return
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves plot
        ax1 = axes[0]
        ax1.plot(epochs, train_losses, 'b-', lw=2, label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', lw=2, label='Validation Loss')
        ax1.set_xlabel('Boosting Round', fontsize=12)
        ax1.set_ylabel(f'Loss ({metric_name})', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Mark best iteration
        if hasattr(self.model, 'best_iteration'):
            best_iter = self.model.best_iteration
            ax1.axvline(x=best_iter + 1, color='green', linestyle='--', lw=2, 
                       label=f'Best iteration: {best_iter}')
            ax1.legend(fontsize=11)
        
        # Loss difference (overfitting indicator)
        ax2 = axes[1]
        loss_diff = np.array(val_losses) - np.array(train_losses)
        ax2.plot(epochs, loss_diff, 'purple', lw=2)
        ax2.axhline(y=0, color='black', linestyle='--', lw=1)
        ax2.fill_between(epochs, 0, loss_diff, where=(loss_diff > 0), 
                        color='red', alpha=0.3, label='Overfitting region')
        ax2.fill_between(epochs, 0, loss_diff, where=(loss_diff <= 0), 
                        color='green', alpha=0.3, label='Underfitting region')
        ax2.set_xlabel('Boosting Round', fontsize=12)
        ax2.set_ylabel('Val Loss - Train Loss', fontsize=12)
        ax2.set_title('Generalization Gap (Overfitting Indicator)', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300)
        plt.close()
        logger.info(f"Saved training curves to {output_dir / 'training_curves.png'}")
        
        # Log final training stats
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        gap = final_val_loss - final_train_loss
        
        logger.info(f"Final training loss: {final_train_loss:.4f}")
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        logger.info(f"Generalization gap: {gap:.4f}")
        
        if gap > 0.1:
            logger.warning(f"⚠️ Large generalization gap ({gap:.4f}) - possible overfitting!")
    
    def _plot_diagnostic_summary(self, metrics: dict, output_dir: Path):
        """Create a summary diagnostic plot with multiple panels"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Center', 'Edge'],
                    yticklabels=['Center', 'Edge'])
        ax1.set_title('Confusion Matrix', fontsize=12)
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. ROC Curve (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(metrics['fpr'], metrics['tpr'], 'darkorange', lw=2, 
                 label=f'AUC = {metrics["roc_auc"]:.4f}')
        ax2.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(metrics['recall_curve'], metrics['precision_curve'], 'green', lw=2,
                 label=f'AP = {metrics["avg_precision"]:.4f}')
        ax3.axhline(y=0.5, color='navy', lw=2, linestyle='--')
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1.05])
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve', fontsize=12)
        ax3.legend(loc='lower left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Score Distributions (middle row, spans 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        y_test = metrics['y_test']
        y_pred_proba = metrics['y_pred_proba']
        center_scores = y_pred_proba[y_test == 0]
        edge_scores = y_pred_proba[y_test == 1]
        ax4.hist(center_scores, bins=30, alpha=0.6, color='blue', label='Center', density=True)
        ax4.hist(edge_scores, bins=30, alpha=0.6, color='red', label='Edge', density=True)
        ax4.axvline(x=0.5, color='black', linestyle='--', lw=2)
        ax4.set_xlabel('Predicted Probability (Edge)')
        ax4.set_ylabel('Density')
        ax4.set_title('Score Distributions', fontsize=12)
        ax4.legend()
        ax4.set_xlim([0, 1])
        ax4.grid(True, alpha=0.3)
        
        # 5. Metrics Summary (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        metrics_text = (
            f"Performance Metrics\n"
            f"{'='*25}\n\n"
            f"Accuracy:  {metrics['accuracy']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall:    {metrics['recall']:.4f}\n"
            f"F1 Score:  {metrics['f1']:.4f}\n\n"
            f"ROC AUC:   {metrics['roc_auc']:.4f}\n"
            f"Avg Prec:  {metrics['avg_precision']:.4f}\n\n"
            f"{'='*25}\n"
            f"Score Statistics\n"
            f"{'='*25}\n\n"
            f"Center μ: {center_scores.mean():.4f}\n"
            f"Center σ: {center_scores.std():.4f}\n"
            f"Edge μ:   {edge_scores.mean():.4f}\n"
            f"Edge σ:   {edge_scores.std():.4f}"
        )
        ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 6. Training Curves (bottom row, if available)
        if self.evals_result:
            metric_name = self.config.eval_metric
            train_losses = self.evals_result.get('validation_0', {}).get(metric_name, [])
            val_losses = self.evals_result.get('validation_1', {}).get(metric_name, [])
            
            if train_losses and val_losses:
                ax6 = fig.add_subplot(gs[2, :2])
                epochs = range(1, len(train_losses) + 1)
                ax6.plot(epochs, train_losses, 'b-', lw=2, label='Training')
                ax6.plot(epochs, val_losses, 'r-', lw=2, label='Validation')
                ax6.set_xlabel('Boosting Round')
                ax6.set_ylabel(f'Loss ({metric_name})')
                ax6.set_title('Training Progress', fontsize=12)
                ax6.legend()
                ax6.grid(True, alpha=0.3)
                
                # Overfitting check text
                ax7 = fig.add_subplot(gs[2, 2])
                ax7.axis('off')
                
                gap = val_losses[-1] - train_losses[-1]
                status = "✓ Good" if abs(gap) < 0.05 else ("⚠️ Overfitting" if gap > 0 else "⚠️ Underfitting")
                
                overfit_text = (
                    f"Training Analysis\n"
                    f"{'='*20}\n\n"
                    f"Train Loss: {train_losses[-1]:.4f}\n"
                    f"Val Loss:   {val_losses[-1]:.4f}\n"
                    f"Gap:        {gap:.4f}\n\n"
                    f"Status: {status}"
                )
                ax7.text(0.1, 0.9, overfit_text, transform=ax7.transAxes, fontsize=11,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Model Diagnostic Summary', fontsize=16, fontweight='bold')
        plt.savefig(output_dir / 'diagnostic_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved diagnostic summary to {output_dir / 'diagnostic_summary.png'}")


def run_classification_pipeline(
    file_pos1: str,
    file_pos2: str,
    config: Optional[IntegrationConfig] = None,
    output_dir: Optional[str] = None
) -> dict:
    """
    Complete pipeline: load traces, integrate, train classifier, evaluate.
    
    Uses trapezoidal integration (proper numerical calculus integration).
    XGBoost is trained with BCE loss (binary:logistic objective).
    
    Args:
        file_pos1: Path to H5 file for position 1
        file_pos2: Path to H5 file for position 2
        config: Integration configuration
        output_dir: Directory to save results and plots
    
    Returns:
        Dictionary of evaluation metrics
    """
    if config is None:
        config = IntegrationConfig()
    
    # Initialize components
    integrator = TraceIntegrator(config)
    classifier = PositionClassifier(config)
    
    # Load and integrate traces from both positions
    logger.info("=" * 60)
    logger.info("LOADING AND INTEGRATING POSITION 1")
    logger.info("=" * 60)
    features_pos1, positions_pos1, stats_pos1 = integrator.load_and_integrate_from_h5(
        file_pos1,
        max_traces=config.n_traces,
        position_label="Position 1"
    )
    
    logger.info("=" * 60)
    logger.info("LOADING AND INTEGRATING POSITION 2")
    logger.info("=" * 60)
    features_pos2, positions_pos2, stats_pos2 = integrator.load_and_integrate_from_h5(
        file_pos2,
        max_traces=config.n_traces,
        position_label="Position 2"
    )
    
    # Compare positions
    logger.info("=" * 60)
    logger.info("POSITION COMPARISON:")
    mean_diff = np.abs(stats_pos1['mean'] - stats_pos2['mean'])
    logger.info(f"  Mean position difference (x, y, z): {mean_diff}")
    logger.info(f"  Position 1 range (x, y, z): {stats_pos1['range']}")
    logger.info(f"  Position 2 range (x, y, z): {stats_pos2['range']}")
    logger.info("=" * 60)
    
    # Prepare dataset
    X, y = classifier.prepare_binary_dataset(features_pos1, features_pos2)
    
    # Train/val/test split
    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )
    
    # Second split: separate validation from training
    val_size_adjusted = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_adjusted,
        random_state=config.random_state,
        stratify=y_trainval
    )
    
    logger.info("=" * 60)
    logger.info("DATASET SPLITS:")
    logger.info(f"  Training:   {len(X_train)} samples")
    logger.info(f"  Validation: {len(X_val)} samples")
    logger.info(f"  Test:       {len(X_test)} samples")
    logger.info("=" * 60)
    
    # Train classifier with validation monitoring
    classifier.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    logger.info("=" * 60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 60)
    output_path = Path(output_dir) if output_dir else None
    metrics = classifier.evaluate(X_test, y_test, output_dir=output_path)
    
    return metrics


def run_center_edge_classification(
    config: Optional[IntegrationConfig] = None,
    output_dir: Optional[str] = None
) -> dict:
    """
    Complete pipeline for center vs edge classification.
    
    Automatically discovers H5 files for the specified energy in center and edge
    folders, loads traces, integrates each channel, and trains XGBoost classifier.
    
    Center positions are labeled as 0, edge positions as 1.
    
    Uses trapezoidal integration (proper numerical calculus integration).
    XGBoost is trained with BCE loss (binary:logistic objective).
    
    Args:
        config: Integration configuration with data paths and energy selection
        output_dir: Directory to save results and plots
    
    Returns:
        Dictionary of evaluation metrics
    """
    if config is None:
        config = IntegrationConfig()
    
    # Initialize components
    integrator = TraceIntegrator(config)
    classifier = PositionClassifier(config)
    
    # Build paths to center and edge folders
    base_path = Path(config.base_data_path)
    center_path = base_path / config.center_folder
    edge_path = base_path / config.edge_folder
    
    logger.info("=" * 60)
    logger.info(f"CENTER vs EDGE CLASSIFICATION")
    logger.info(f"Base data path: {base_path}")
    logger.info(f"Energy: {config.energy}")
    logger.info(f"Recoil types: {config.recoil_types}")
    logger.info("=" * 60)
    
    # Discover H5 files for center and edge
    logger.info("\n" + "=" * 60)
    logger.info("DISCOVERING CENTER FILES")
    logger.info("=" * 60)
    center_files = integrator.discover_h5_files(
        str(center_path),
        config.energy,
        config.recoil_types
    )
    
    if not center_files:
        raise ValueError(f"No center files found for energy={config.energy}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DISCOVERING EDGE FILES")
    logger.info("=" * 60)
    edge_files = integrator.discover_h5_files(
        str(edge_path),
        config.energy,
        config.recoil_types
    )
    
    if not edge_files:
        raise ValueError(f"No edge files found for energy={config.energy}")
    
    # Load and integrate traces from center (label=0)
    logger.info("\n" + "=" * 60)
    logger.info("LOADING AND INTEGRATING CENTER TRACES (label=0)")
    logger.info("=" * 60)
    features_center, positions_center, stats_center = integrator.load_and_integrate_from_multiple_h5(
        center_files,
        max_traces=config.n_traces,
        position_label="Center"
    )
    
    # Load and integrate traces from edge (label=1)
    logger.info("\n" + "=" * 60)
    logger.info("LOADING AND INTEGRATING EDGE TRACES (label=1)")
    logger.info("=" * 60)
    features_edge, positions_edge, stats_edge = integrator.load_and_integrate_from_multiple_h5(
        edge_files,
        max_traces=config.n_traces,
        position_label="Edge"
    )
    
    # Compare positions
    logger.info("\n" + "=" * 60)
    logger.info("POSITION COMPARISON:")
    mean_diff = np.abs(stats_center['mean'] - stats_edge['mean'])
    logger.info(f"  Mean position difference (x, y, z): {mean_diff}")
    logger.info(f"  Center mean (x, y, z): {stats_center['mean']}")
    logger.info(f"  Edge mean (x, y, z):   {stats_edge['mean']}")
    logger.info(f"  Center range (x, y, z): {stats_center['range']}")
    logger.info(f"  Edge range (x, y, z):   {stats_edge['range']}")
    logger.info("=" * 60)
    
    # Prepare dataset (center=0, edge=1)
    X, y = classifier.prepare_binary_dataset(features_center, features_edge)
    
    logger.info(f"\nTotal samples: {len(X)} (Center: {len(features_center)}, Edge: {len(features_edge)})")
    logger.info(f"Feature dimension: {X.shape[1]} (integrated channel energies)")
    
    # Train/val/test split
    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )
    
    # Second split: separate validation from training
    val_size_adjusted = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_adjusted,
        random_state=config.random_state,
        stratify=y_trainval
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("DATASET SPLITS:")
    logger.info(f"  Training:   {len(X_train)} samples")
    logger.info(f"  Validation: {len(X_val)} samples")
    logger.info(f"  Test:       {len(X_test)} samples")
    logger.info("=" * 60)
    
    # Train classifier with validation monitoring
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING XGBOOST CLASSIFIER")
    logger.info("=" * 60)
    classifier.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 60)
    output_path = Path(output_dir) if output_dir else None
    metrics = classifier.evaluate(X_test, y_test, output_dir=output_path)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration-based position classification using trapezoidal integration")
    parser.add_argument("--file1", type=str, required=True, help="H5 file for position 1")
    parser.add_argument("--file2", type=str, required=True, help="H5 file for position 2")
    parser.add_argument("--n-traces", type=int, default=1000, help="Number of traces per position")
    parser.add_argument("--output-dir", type=str, default="./integration_results", help="Output directory")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of boosting rounds (like epochs)")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    
    args = parser.parse_args()
    
    config = IntegrationConfig(
        n_traces=args.n_traces,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )
    
    metrics = run_classification_pipeline(
        file_pos1=args.file1,
        file_pos2=args.file2,
        config=config,
        output_dir=args.output_dir
    )
