"""Evaluate model performance"""

import math
import logging

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score

from reconstruction_model.utils import get_next_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note this function will contain other metrics later i.e. F1, accuracy etc
# when ER/NR classification is included.
@torch.no_grad()
def eval_model(
    model,
    generator,
    device,
    config,
    target_normaliser=None,
    energy_normaliser=None,
):
    model_was_training = model.training
    num_eval_steps = config.num_eval_steps
    model.eval()
    # Accumulate loss (GPU)

    val_total_loss = 0.
    val_total_spatial_loss = 0.
    val_total_energy_loss = 0
    val_total_class_loss = 0.
    class_preds_arr = []
    class_targets_arr = []
    # Accumulate denormalised RMSE (CPU)
    val_total_spatial_sq_errors = 0.
    val_total_energy_sq_errors = 0.

    for step in range(num_eval_steps):
        val_inputs, val_spatial_targets, val_energy_targets, val_class_targets = get_next_batch(
            generator,
            device,
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            val_spatial_logits, val_energy_logits, val_class_logits = model(val_inputs)

        val_spatial_loss = F.mse_loss(
            val_spatial_logits,
            val_spatial_targets,
            reduction="mean",
        )
        val_energy_loss = F.mse_loss(
            val_energy_logits,
            val_energy_targets.unsqueeze(-1),
            reduction="mean",
        )
        val_class_loss = F.binary_cross_entropy_with_logits(
            val_class_logits,
            val_class_targets.unsqueeze(-1),
            reduction="mean",
        )
        val_step_total_loss = (
            config.scalar_loss_weights[0] * val_spatial_loss
            + config.scalar_loss_weights[1] * val_energy_loss
            + config.scalar_loss_weights[2] * val_class_loss
        )
        # Accumulate validation loss
        val_total_spatial_loss += val_spatial_loss.item()
        val_total_energy_loss += val_energy_loss.item()
        val_total_class_loss += val_class_loss.item()
        val_total_loss += val_step_total_loss.item()
        # Denormalise and accumulate squared errors
        if target_normaliser is not None:
            # Position: denormalize from z-score
            val_spatial_pred_denorm = target_normaliser.denormalise(
                val_spatial_logits.cpu().float().numpy()
            )
            val_spatial_true_denorm = target_normaliser.denormalise(
                val_spatial_targets.cpu().float().numpy()
            )
            # Compute and accumulate square errors
            val_spatial_errors = val_spatial_pred_denorm - val_spatial_true_denorm
            val_total_spatial_sq_errors += np.sum(val_spatial_errors ** 2)
    
        # Energy: denormalise energies
        if energy_normaliser is not None:
            val_energy_pred_denorm = energy_normaliser.denormalise(
                val_energy_logits.cpu().float().numpy()
            )
            val_energy_true_denorm = energy_normaliser.denormalise(
                val_energy_targets.unsqueeze(-1).cpu().float().numpy()
            )            
            val_energy_errors = val_energy_pred_denorm - val_energy_true_denorm
            val_total_energy_sq_errors += np.sum(val_energy_errors ** 2)
        
        class_probs = torch.sigmoid(val_class_logits.squeeze(-1))
        class_preds_arr.extend((class_probs > 0.5).cpu().numpy())
        class_targets_arr.extend(val_class_targets.cpu().numpy())

    # Normalise accumulated metrics by number of eval steps:
    val_total_spatial_loss /= num_eval_steps
    val_total_energy_loss /= num_eval_steps
    val_total_class_loss /= num_eval_steps
    val_total_loss /= num_eval_steps
    val_total_spatial_sq_errors /= (num_eval_steps * config.device_batch_size)
    val_total_energy_sq_errors /= (num_eval_steps * config.device_batch_size)
    
    if target_normaliser is not None:
        val_spatial_rmse = np.sqrt(val_total_spatial_sq_errors)
    else:
        val_spatial_rmse = math.sqrt(val_total_spatial_loss)
    
    if energy_normaliser is not None:
        val_energy_rmse = np.sqrt(val_total_energy_sq_errors)
    else:
        val_energy_rmse = math.sqrt(val_total_energy_loss)
    
    val_accuracy = np.mean(np.array(class_preds_arr) == np.array(class_targets_arr))
    model.train(model_was_training)
    val_metrics = dict(
        val_total_loss=val_total_loss,
        val_total_spatial_loss=val_total_spatial_loss,
        val_total_energy_loss=val_total_energy_loss,
        val_total_class_loss=val_total_class_loss,
        val_spatial_rmse=val_spatial_rmse,
        val_energy_rmse=val_energy_rmse,
        val_accuracy=val_accuracy,
    )
    return val_metrics

def compute_performance_metrics(
    spatial_preds,
    spatial_targets,
    energy_preds,
    energy_targets,
    metrics,
    metric_type,
    compute_r2_score=True,
    energy_bin=None,
):  
    # Spatial_metrics
    spatial_errors = spatial_preds - spatial_targets
    metrics[f"{metric_type}_spatial_rmse"] = np.sqrt(np.mean(spatial_errors ** 2)).item()
    metrics[f"{metric_type}_spatial_mae"] = np.mean(np.abs(spatial_errors)).item()
    
    # Per coordinate metrics
    for i, coord in enumerate(["x", "y"]):
        metrics[f"{metric_type}_{coord}_rmse"] = np.sqrt(np.mean(spatial_errors[:, i] ** 2)).item()
        metrics[f"{metric_type}_{coord}_mae"] = np.mean(np.abs(spatial_errors[:, i])).item()
    
    # 3D distance error
    metrics[f"{metric_type}_mean_euclidean_distance"] = np.mean(np.linalg.norm(spatial_errors, axis=1)).item()

    # Energy metrics
    energy_errors = energy_preds - energy_targets
    relative_energy_errors = energy_errors / energy_targets
    metrics[f"{metric_type}_energy_rmse"] = np.sqrt(np.mean(energy_errors ** 2)).item()
    metrics[f"{metric_type}_energy_mae"] = np.mean(np.abs(energy_errors)).item()
    metrics[f"{metric_type}_energy_resolution"] = (np.std(energy_preds) / np.mean(energy_preds)).item()
    if energy_bin is not None:
        metrics[f"{metric_type}_energy_bias"] = ((np.mean(energy_preds) - energy_bin) / energy_bin).item()

    if compute_r2_score is True:
        metrics[f"{metric_type}_spatial_r2"] = r2_score(
            spatial_targets,
            spatial_preds,
            multioutput="raw_values",
        ).tolist()
        metrics[f"{metric_type}_energy_r2"] = r2_score(
            energy_targets,
            energy_preds,
        )
    metrics[f"{metric_type}_n_samples"] = len(spatial_preds)
    return metrics

@torch.no_grad()
def evaluate_test_set(
    dataloader,
    model,
    device,
    target_normaliser=None,
    energy_normaliser=None,
    energy_bins=None,
):  
    spatial_preds_arr = []
    spatial_targets_arr = []
    energy_preds_arr = []
    energy_targets_arr = []
    for batch in dataloader:
        inputs, spatial_targets, energy_targets, _ = batch
        inputs = inputs.to(device)
        spatial_targets = spatial_targets.to(device)
        # (B,) -> (B, 1)
        energy_targets = energy_targets.unsqueeze(-1).to(device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            spatial_preds, energy_preds, class_logits = model(inputs)

        if target_normaliser is not None:
            spatial_preds_denorm = target_normaliser.denormalise(
                spatial_preds.cpu().float().numpy(), 
            )
            spatial_targets_denorm = target_normaliser.denormalise(
                spatial_targets.cpu().float().numpy(),
            )
        else:
            spatial_preds_denorm = spatial_preds.cpu().float().numpy()
            spatial_targets_denorm = spatial_targets.cpu().float().numpy()
        
        if energy_normaliser is not None:
            energy_preds_denorm = energy_normaliser.denormalise(
                energy_preds.cpu().float().numpy(),
            )
            energy_targets_denorm = energy_normaliser.denormalise(
                energy_targets.cpu().float().numpy(),
            )
        else:
            energy_preds_denorm = energy_preds.cpu().float().numpy()
            energy_targets_denorm = energy_targets.cpu().float().numpy()
        
        spatial_preds_arr.append(spatial_preds_denorm)
        spatial_targets_arr.append(spatial_targets_denorm)
        energy_preds_arr.append(energy_preds_denorm)
        energy_targets_arr.append(energy_targets_denorm)

    # Concatenate batches
    # (num_batches * B, 3)
    spatial_preds_combined = np.concatenate(spatial_preds_arr, axis=0)
    spatial_targets_combined = np.concatenate(spatial_targets_arr, axis=0)
    # (num_batches * B, 1)
    energy_preds_combined = np.concatenate(energy_preds_arr, axis=0)
    energy_targets_combined = np.concatenate(energy_targets_arr, axis=0)

    metrics = {}
    # Compute overall metrics
    logger.info("Computing overall performance metrics...")
    compute_r2_score = len(spatial_preds_combined) > 1
    metrics = compute_performance_metrics(
        spatial_preds_combined,
        spatial_targets_combined,
        energy_preds_combined,
        energy_targets_combined,
        metrics,
        metric_type="overall",
        compute_r2_score=len(spatial_preds_combined) > 1,
    )
    if energy_bins is not None:
        logger.info(f"Computing metrics for energy_bins: {energy_bins} eV")
        for energy in energy_bins:
            metrics_prefix = f"{energy}_eV"
            mask = np.isclose(energy_targets_combined.squeeze(), energy)
            metrics = compute_performance_metrics(
                spatial_preds_combined[mask, :],
                spatial_targets_combined[mask, :],
                energy_preds_combined[mask, :],
                energy_targets_combined[mask, :],
                metrics,
                metric_type=metrics_prefix,
                compute_r2_score=len(spatial_preds_combined[mask, :]) > 1,
                energy_bin=energy,
            )
    
    return metrics