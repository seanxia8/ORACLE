"""
visualise detector geometry, true event positions, and model predictions in 3D.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from reconstruction_model.model import Transformer, TransformerConfig
from reconstruction_model.dataset_temp import DataConfig, create_dataloaders
from reconstruction_model.checkpoints import load_checkpoint


try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Plotly is required for visualization. Install with: pip install plotly")


def load_detector_positions(position_file: Path) -> np.ndarray:
    """Load detector positions from position_MMC_V2.dat file."""
    positions = []
    with open(position_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                positions.append([x, y, z])
    
    positions = np.array(positions)
    print(f"Loaded {len(positions)} detector positions")
    print(f"Position range: X=[{positions[:,0].min():.1f}, {positions[:,0].max():.1f}] mm")
    print(f"                Y=[{positions[:,1].min():.1f}, {positions[:,1].max():.1f}] mm")
    print(f"                Z=[{positions[:,2].min():.1f}, {positions[:,2].max():.1f}] mm")
    return positions


def load_model_and_predict(
    model_path: Path,
    data_config: DataConfig,
    sample_indices: list[int],
    split: str = "test",
    batch_size: int = 32
) -> Tuple[list, list, list, list, list]:
    """Load a trained model and make predictions on multiple test samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"\nCreating dataloaders with pre-computed normalization stats...")
    
    # Create dataloaders - use cached position normalization only
    dataloaders = create_dataloaders(
        data_config=data_config,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 for simplicity in visualization
        max_samples_for_norm=1000,  # Not used when loading from cache
        precomputed_trace=False,     # Not normalizing inputs
        precomputed_positions=True,  # Load cached position normalization
        precomputed_energy=False     # Not normalizing energy
    )
    
    # Get the test dataloader and access its dataset
    test_loader = dataloaders[split]
    dataset = test_loader.dataset
    print(f"Dataset size ({split}): {len(dataset)}")
    
    config = TransformerConfig()
    model = Transformer(config).to(device)
    
    print(f"\nLoading checkpoint from {model_path}")
    model, _, _ = load_checkpoint(
        model = model,
        model_path = model_path, 
        device = device,
    )
    model.eval()
    print("Model loaded successfully")
    
    # Store results for all samples
    true_positions = []
    pred_positions = []
    true_energies = []
    pred_energies = []
    recoil_types = []
    
    print(f"\nProcessing {len(sample_indices)} samples...")
    for i, sample_idx in enumerate(sample_indices):
        print(f"\n[{i+1}/{len(sample_indices)}] Sample {sample_idx}...")
        input_trace, spatial_target, energy_target, recoil_type = dataset[sample_idx]

        input_trace = input_trace.unsqueeze(0).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                spatial_pred, energy_pred = model(input_trace)
        
        # Convert from bfloat16 to float32 before numpy conversion
        pred_position = spatial_pred.squeeze(0).float().cpu().numpy()
        true_position = spatial_target.numpy()
        pred_energy = energy_pred.squeeze(0).float().cpu().numpy()[0]
        true_energy = energy_target.numpy()
        
        # denormalise positions if normalisation was applied
        if data_config.normalise_positions and dataset.target_normaliser is not None:
            pred_position = dataset.target_normaliser.denormalise(pred_position)
            true_position = dataset.target_normaliser.denormalise(true_position)
        
        # denormalise energy if normalisation was applied
        if data_config.normalise_energy and dataset.energy_normaliser is not None:
            pred_energy = dataset.energy_normaliser.denormalise(pred_energy)
            true_energy = dataset.energy_normaliser.denormalise(true_energy)
        
        # Store results
        true_positions.append(true_position)
        pred_positions.append(pred_position)
        true_energies.append(true_energy)
        pred_energies.append(pred_energy)
        recoil_types.append(recoil_type)
        
        # Print summary
        spatial_error = np.linalg.norm(pred_position - true_position)
        print(f"  True: [{true_position[0]:.1f}, {true_position[1]:.1f}, {true_position[2]:.1f}] mm")
        print(f"  Pred: [{pred_position[0]:.1f}, {pred_position[1]:.1f}, {pred_position[2]:.1f}] mm")
        print(f"  Error: {spatial_error:.2f} mm | Type: {recoil_type}")
    
    return true_positions, pred_positions, true_energies, pred_energies, recoil_types


def plot_with_plotly(
    detector_positions: np.ndarray,
    true_positions: list[np.ndarray],
    pred_positions: list[np.ndarray],
    energies: list[float] = None,
    recoil_types: list[str] = None,
    save_path: Optional[Path] = None,
    title_suffix: str = ""
):
    """Create interactive 3D plot using Plotly with multiple event points."""
    
    # Get detector z range for filtering
    z_min = detector_positions[:, 2].min()
    z_max = detector_positions[:, 2].max()
    
    top_detectors = detector_positions[:19]
    bottom_detectors = detector_positions[19:]
    
    traces = []
    
    # All Detectors (same color and shape)
    all_detector_x = np.concatenate([top_detectors[:, 0], bottom_detectors[:, 0]])
    all_detector_y = np.concatenate([top_detectors[:, 1], bottom_detectors[:, 1]])
    all_detector_z = np.concatenate([top_detectors[:, 2], bottom_detectors[:, 2]])
    all_detector_text = [f'Channel {i}' for i in range(len(detector_positions))]
    
    traces.append(go.Scatter3d(
        x=all_detector_x, y=all_detector_y, z=all_detector_z,
        mode='markers',
        marker=dict(size=8, color='lightblue', symbol='circle', 
                   line=dict(color='blue', width=2)),
        name='Detectors',
        text=all_detector_text,
        hoverinfo='text'
    ))
    
    # Separate positions by recoil type and filter by z range
    true_nr_x, true_nr_y, true_nr_z, true_nr_texts = [], [], [], []
    true_er_x, true_er_y, true_er_z, true_er_texts = [], [], [], []
    pred_nr_x, pred_nr_y, pred_nr_z, pred_nr_texts = [], [], [], []
    pred_er_x, pred_er_y, pred_er_z, pred_er_texts = [], [], [], []
    spatial_errors = []
    filtered_count = 0
    
    for i, (true_pos, pred_pos) in enumerate(zip(true_positions, pred_positions)):
        # Filter out positions outside detector z range
        if true_pos[2] < z_min or true_pos[2] > z_max:
            filtered_count += 1
            continue
            
        error = np.linalg.norm(pred_pos - true_pos)
        spatial_errors.append(error)
        
        recoil = recoil_types[i] if recoil_types else "Unknown"
        true_text = f'Sample {i} (True {recoil})<br>True: ({true_pos[0]:.1f}, {true_pos[1]:.1f}, {true_pos[2]:.1f})<br>Error: {error:.1f} mm'
        pred_text = f'Sample {i} (Pred {recoil})<br>Pred: ({pred_pos[0]:.1f}, {pred_pos[1]:.1f}, {pred_pos[2]:.1f})<br>Error: {error:.1f} mm'
        
        # Separate by recoil type
        if recoil == "NR":
            true_nr_x.append(true_pos[0])
            true_nr_y.append(true_pos[1])
            true_nr_z.append(true_pos[2])
            true_nr_texts.append(true_text)
            
            pred_nr_x.append(pred_pos[0])
            pred_nr_y.append(pred_pos[1])
            pred_nr_z.append(pred_pos[2])
            pred_nr_texts.append(pred_text)
        else:  # ER
            true_er_x.append(true_pos[0])
            true_er_y.append(true_pos[1])
            true_er_z.append(true_pos[2])
            true_er_texts.append(true_text)
            
            pred_er_x.append(pred_pos[0])
            pred_er_y.append(pred_pos[1])
            pred_er_z.append(pred_pos[2])
            pred_er_texts.append(pred_text)
        
        # Add error line for each sample
        traces.append(go.Scatter3d(
            x=[true_pos[0], pred_pos[0]],
            y=[true_pos[1], pred_pos[1]],
            z=[true_pos[2], pred_pos[2]],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add True NR positions (green squares)
    if true_nr_x:
        traces.append(go.Scatter3d(
            x=true_nr_x, y=true_nr_y, z=true_nr_z,
            mode='markers',
            marker=dict(size=12, color='green', symbol='square',
                       line=dict(color='darkgreen', width=2)),
            name='True NR',
            text=true_nr_texts,
            hoverinfo='text'
        ))
    
    # Add Pred NR positions (green crosses)
    if pred_nr_x:
        traces.append(go.Scatter3d(
            x=pred_nr_x, y=pred_nr_y, z=pred_nr_z,
            mode='markers',
            marker=dict(size=12, color='green', symbol='x',
                       line=dict(color='darkgreen', width=2)),
            name='Pred NR',
            text=pred_nr_texts,
            hoverinfo='text'
        ))
    
    # Add True ER positions (orange squares)
    if true_er_x:
        traces.append(go.Scatter3d(
            x=true_er_x, y=true_er_y, z=true_er_z,
            mode='markers',
            marker=dict(size=12, color='orange', symbol='square',
                       line=dict(color='darkorange', width=2)),
            name='True ER',
            text=true_er_texts,
            hoverinfo='text'
        ))
    
    # Add Pred ER positions (orange crosses)
    if pred_er_x:
        traces.append(go.Scatter3d(
            x=pred_er_x, y=pred_er_y, z=pred_er_z,
            mode='markers',
            marker=dict(size=12, color='orange', symbol='x',
                       line=dict(color='darkorange', width=2)),
            name='Pred ER',
            text=pred_er_texts,
            hoverinfo='text'
        ))
    
    mean_spatial_error = np.mean(spatial_errors) if spatial_errors else 0.0
    
    title_parts = [f'DELight Detector Geometry ({len(spatial_errors)} samples']
    if filtered_count > 0:
        title_parts[0] += f', {filtered_count} filtered'
    title_parts[0] += ')'
    if energies and energies[0] is not None:
        title_parts.append(f'Energy: {energies[0]:.0f} eV')
    title_parts.append(f'Mean Spatial Error: {mean_spatial_error:.2f} mm')
    if title_suffix:
        title_parts.append(title_suffix)
    
    title = '<br>'.join([title_parts[0], ' | '.join(title_parts[1:])])
    
    layout = go.Layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=True,
        hovermode='closest',
        width=1400,
        height=1000
    )
    
    fig = go.Figure(data=traces, layout=layout)
    
    # Save to HTML file instead of trying to open in remote browser
    if save_path is None:
        save_path = Path("detector_geometry_visualisation.html")
    else:
        save_path = save_path.with_suffix('.html')
    
    print("\n" + "="*60)
    print(f"Saving interactive 3D plot to: {save_path}")
    
    # Use CDN for smaller file size and faster loading
    fig.write_html(
        str(save_path),
        config={'displayModeBar': True, 'responsive': True},
        include_plotlyjs='cdn'
    )
    
    print(f"✓ Saved! File location:")
    print(f"  {save_path.absolute()}")
    print("\nControls once opened:")
    print("  - Click and drag to rotate")
    print("  - Scroll to zoom")
    print("  - Hover over points for details")
    print("="*60)
    
    # Also save static PNG if kaleido is available
    png_path = save_path.with_suffix('.png')
    try:
        fig.write_image(str(png_path), width=1600, height=1200)
        print(f"\n✓ Also saved static image to: {png_path}")
    except:
        pass


def main(n_samples: int = 10, 
         TEST_MODE: bool = False,
         CLASSIFY: bool = False):
    """Main function - configure your settings here!"""
    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    if TEST_MODE:
        print("="*60)
        print("TEST MODE - Using custom coordinates")
        print("="*60)
        
        true_position = np.array([0.0, 0.0, -1700.0])
        pred_position = np.array([20.0, 15.0, -1695.0])
        energy = 500
        recoil_type = "TEST"
        
        print(f"\nTest positions:")
        print(f"True: {true_position}")
        print(f"Pred: {pred_position}")
        print(f"Error: {np.linalg.norm(pred_position - true_position):.2f} mm")

    elif CLASSIFY:
        print("="*60)
        print("CLASSIFICATION MODE - Plotting centre and edge positions")
        print("="*60)
        
        true_position = np.array([0.0, 0.0, -1700.0])
        pred_position = np.array([0.0, -125.0, -1700.0])
        energy = 500
        recoil_type = "TEST"

    else:
        model_path = Path("training_checkpoints/reconstruction_model_10000.pt")  
        
        # Set target_energy to filter dataset by specific energy (much faster!)
        target_energy = 500  # Set to None to load all energies
        
        data_config = DataConfig(
            access_mode="remote",
            remote_data_path="/ceph/srv/ssjostrom/training_small",
            recoil_types=["NR"],  # Must be a list
            normalise_positions=True,   # Only normalize positions
            normalise_energy=False,     # Don't normalize energy
            normalise_inputs=False,     # Don't normalize inputs
            target_energy=target_energy,
            normalisation_mode='zscore',  # Use z-score normalization
            # Use cached position normalization stats
            position_norm_cache="/ceph/lwindett/DELight_reconstruction/cache/position_zscore_500eV.pkl",
        )
        
        split = "test"
        
        # Generate random indices (will be validated against dataset size)
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.randint(0, 100, size=n_samples).tolist()  
    
    position_file = Path("position_MMC_V2.dat")
    save_path = Path("detector_geometry_visualisation.html")
    
    # ========================================
    # VISUALISATION
    # ========================================
    
    print("\nLoading detector positions...")
    detector_positions = load_detector_positions(position_file)
    
    if TEST_MODE or CLASSIFY:
        plot_with_plotly(
            detector_positions, [true_position], [pred_position],
            energies=[energy], recoil_types=[recoil_type],
            save_path=save_path,
            title_suffix="(Test Mode)"
        )
    else:
        true_positions, pred_positions, true_energies, pred_energies, recoil_types = load_model_and_predict(
            model_path, data_config, sample_indices=sample_indices, split=split
        )
        plot_with_plotly(
            detector_positions, true_positions, pred_positions,
            energies=true_energies, recoil_types=recoil_types,
            save_path=save_path,
            title_suffix=f"Random {n_samples} samples"
        )


def plot_all_true_positions(
    model_path: Path,
    data_config: DataConfig,
    split: str = "test",
    save_path: Optional[Path] = None
):
    """Load all true positions from the test dataset and visualize them."""
    print("\n" + "="*60)
    print("PLOTTING ALL TRUE POSITIONS FROM DATASET")
    print("="*60)
    
    # Create dataloaders
    print(f"\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        data_config=data_config,
        batch_size=32,
        num_workers=0,
        max_samples_for_norm=1000,
        precomputed_trace=False,      # Not normalizing inputs
        precomputed_positions=True,   # Use cached normalization
        precomputed_energy=False      # Not normalizing energy
    )
    
    test_loader = dataloaders[split]
    dataset = test_loader.dataset
    print(f"Dataset size ({split}): {len(dataset)}")
    
    # Collect all true positions
    true_positions = []
    true_energies = []
    recoil_types = []
    
    print(f"\nCollecting all {len(dataset)} true positions...")
    for i in range(len(dataset)):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        
        _, spatial_target, energy_target, recoil_type = dataset[i]
        true_position = spatial_target.numpy()
        true_energy = energy_target.numpy()
        
        # denormalise if needed
        if data_config.normalise_positions and dataset.target_normaliser is not None:
            true_position = dataset.target_normaliser.denormalise(true_position)
        
        if data_config.normalise_energy and dataset.energy_normaliser is not None:
            true_energy = dataset.energy_normaliser.denormalise(true_energy)
        
        true_positions.append(true_position)
        true_energies.append(true_energy)
        recoil_types.append(recoil_type)
    
    print(f"✓ Collected {len(true_positions)} positions")
    
    # Load detector positions
    position_file = Path("position_MMC_V2.dat")
    detector_positions = load_detector_positions(position_file)
    
    # Plot
    if save_path is None:
        save_path = Path("detector_geometry_all_true_positions.html")
    
    plot_true_positions_only(
        detector_positions,
        true_positions,
        energies=true_energies,
        recoil_types=recoil_types,
        save_path=save_path
    )


def plot_true_positions_only(
    detector_positions: np.ndarray,
    true_positions: list[np.ndarray],
    energies: list[float] = None,
    recoil_types: list[str] = None,
    save_path: Path = None
):
    """Create interactive 3D plot showing only true event positions."""
    
    top_detectors = detector_positions[:19]
    bottom_detectors = detector_positions[19:]
    
    traces = []
    
    # Bottom Sensors
    traces.append(go.Scatter3d(
        x=top_detectors[:, 0], y=top_detectors[:, 1], z=top_detectors[:, 2],
        mode='markers',
        marker=dict(size=6, color='lightblue', symbol='circle', 
                   line=dict(color='blue', width=1)),
        name='Bottom Sensors',
        text=[f'Channel {i}' for i in range(19)],
        hoverinfo='text'
    ))
    
    # Top Sensors
    traces.append(go.Scatter3d(
        x=bottom_detectors[:, 0], y=bottom_detectors[:, 1], z=bottom_detectors[:, 2],
        mode='markers',
        marker=dict(size=6, color='lightcoral', symbol='square',
                   line=dict(color='red', width=1)),
        name='Top Sensors',
        text=[f'Channel {i+19}' for i in range(len(bottom_detectors))],
        hoverinfo='text'
    ))
    
    # Separate ER and NR events
    er_positions = []
    nr_positions = []
    
    for i, (true_pos, recoil_type) in enumerate(zip(true_positions, recoil_types)):
        if recoil_type == "ER":
            er_positions.append(true_pos)
        else:
            nr_positions.append(true_pos)
    
    # Plot ER events
    if er_positions:
        er_positions = np.array(er_positions)
        traces.append(go.Scatter3d(
            x=er_positions[:, 0], y=er_positions[:, 1], z=er_positions[:, 2],
            mode='markers',
            marker=dict(size=4, color='green', symbol='circle', opacity=0.6),
            name=f'ER Events (n={len(er_positions)})',
            text=[f'ER Event {i}<br>({er_positions[i,0]:.1f}, {er_positions[i,1]:.1f}, {er_positions[i,2]:.1f})' 
                  for i in range(len(er_positions))],
            hoverinfo='text'
        ))
    
    # Plot NR events
    if nr_positions:
        nr_positions = np.array(nr_positions)
        traces.append(go.Scatter3d(
            x=nr_positions[:, 0], y=nr_positions[:, 1], z=nr_positions[:, 2],
            mode='markers',
            marker=dict(size=4, color='purple', symbol='diamond', opacity=0.6),
            name=f'NR Events (n={len(nr_positions)})',
            text=[f'NR Event {i}<br>({nr_positions[i,0]:.1f}, {nr_positions[i,1]:.1f}, {nr_positions[i,2]:.1f})' 
                  for i in range(len(nr_positions))],
            hoverinfo='text'
        ))
    
    energy_str = f"{energies[0]:.0f} eV" if energies and energies[0] is not None else "Mixed"
    title = f'DELight True Event Positions<br>Energy: {energy_str} | Total Events: {len(true_positions)}'
    
    layout = go.Layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=True,
        hovermode='closest',
        width=1400,
        height=1000
    )
    
    fig = go.Figure(data=traces, layout=layout)
    
    print("\n" + "="*60)
    print(f"Saving true positions plot to: {save_path}")
    
    fig.write_html(
        str(save_path),
        config={'displayModeBar': True, 'responsive': True},
        include_plotlyjs='cdn'
    )
    
    print(f"✓ Saved! File location:")
    print(f"  {save_path.absolute()}")
    print("="*60)


# ============================================================================
# XGBoost Position Regressor Visualization Functions
# ============================================================================

def load_xgboost_model_and_predict(
    model_path: Path,
    data_config: DataConfig,
    n_samples: int = 10,
    split: str = "test",
    batch_size: int = 32,
    integration_dt: float = 2.56e-7,
    random_seed: int = 42
) -> Tuple[list, list, list]:
    """
    Load a trained XGBoost model and make predictions on test samples.
    
    Args:
        model_path: Path to saved XGBoost model directory
        data_config: Data configuration
        n_samples: Number of samples to predict
        split: Data split to use ("train", "val", "test")
        batch_size: Batch size for dataloader
        integration_dt: Time step for trace integration
    
    Returns:
        Tuple of (true_positions, pred_positions, recoil_types)
    """
    from reconstruction_model.model import XGBoostPositionRegressor
    from reconstruction_model.dataset import create_dataloaders
    
    print(f"\nLoading XGBoost model from {model_path}")
    model = XGBoostPositionRegressor.load(model_path)
    print("Model loaded successfully")
    
    print(f"\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        data_config=data_config,
        batch_size=batch_size,
        num_workers=0,
        precomputed_trace=False,  # Don't need trace normalization for XGBoost
        precomputed_positions=data_config.normalise_positions,  # Use if available
        precomputed_energy=False
    )
    
    dataset = dataloaders[split].dataset
    print(f"Dataset size ({split}): {len(dataset)}")

    np.random.seed(random_seed)
    sample_indices = np.random.randint(0, len(dataset), size=n_samples).tolist()
    
    # Validate sample indices
    valid_indices = [i for i in sample_indices if i < len(dataset)]
    if len(valid_indices) < len(sample_indices):
        print(f"Warning: {len(sample_indices) - len(valid_indices)} indices out of range")
    
    true_positions = []
    pred_positions = []
    recoil_types = []
    
    print(f"\nProcessing {len(valid_indices)} samples...")
    for i, sample_idx in enumerate(valid_indices):
        print(f"\n[{i+1}/{len(valid_indices)}] Sample {sample_idx}...")
        
        input_trace, spatial_target, energy_target, recoil_type = dataset[sample_idx]
        
        # Integrate traces
        trace_np = input_trace.numpy()
        integrated = XGBoostPositionRegressor.integrate_traces(trace_np, dt=integration_dt)
        
        # Predict (automatically denormalises if model has target_normaliser)
        pred_position = model.predict(integrated, denormalise=True)
        true_position = spatial_target.numpy()
        
        # denormalise true position if needed
        if data_config.normalise_positions and dataset.target_normaliser is not None:
            true_position = dataset.target_normaliser.denormalise(true_position)
        
        true_positions.append(true_position)
        pred_positions.append(pred_position)
        recoil_types.append(recoil_type)
        
        # Print summary
        spatial_error = np.linalg.norm(pred_position - true_position)
        print(f"  True: [{true_position[0]:.1f}, {true_position[1]:.1f}, {true_position[2]:.1f}] mm")
        print(f"  Pred: [{pred_position[0]:.1f}, {pred_position[1]:.1f}, {pred_position[2]:.1f}] mm")
        print(f"  Error: {spatial_error:.2f} mm | Type: {recoil_type}")
    
    return true_positions, pred_positions, recoil_types


def plot_xgboost_predictions_3d(
    detector_positions: np.ndarray,
    true_positions: list[np.ndarray],
    pred_positions: list[np.ndarray],
    recoil_types: list[str] = None,
    save_path: Optional[Path] = None,
    title_suffix: str = ""
):
    """
    Create interactive 3D plot of XGBoost model predictions using Plotly.
    
    Args:
        detector_positions: Detector positions array
        true_positions: List of true position arrays
        pred_positions: List of predicted position arrays
        recoil_types: List of recoil types
        save_path: Path to save HTML file
        title_suffix: Additional text for title
    """
    # Get detector z range for filtering
    z_min = detector_positions[:, 2].min()
    z_max = detector_positions[:, 2].max()
    
    top_detectors = detector_positions[:19]
    bottom_detectors = detector_positions[19:]
    
    traces = []
    
    # All Detectors (combined with shared legend)
    all_detector_x = np.concatenate([top_detectors[:, 0], bottom_detectors[:, 0]])
    all_detector_y = np.concatenate([top_detectors[:, 1], bottom_detectors[:, 1]])
    all_detector_z = np.concatenate([top_detectors[:, 2], bottom_detectors[:, 2]])
    all_detector_text = [f'Channel {i}' for i in range(len(detector_positions))]
    
    traces.append(go.Scatter3d(
        x=all_detector_x, y=all_detector_y, z=all_detector_z,
        mode='markers',
        marker=dict(size=6, color='lightblue', symbol='circle', 
                   line=dict(color='blue', width=1)),
        name='Detectors',
        text=all_detector_text,
        hoverinfo='text'
    ))
    
    # Separate positions by recoil type and filter by z range
    true_nr_x, true_nr_y, true_nr_z, true_nr_texts = [], [], [], []
    true_er_x, true_er_y, true_er_z, true_er_texts = [], [], [], []
    pred_nr_x, pred_nr_y, pred_nr_z, pred_nr_texts = [], [], [], []
    pred_er_x, pred_er_y, pred_er_z, pred_er_texts = [], [], [], []
    spatial_errors = []
    filtered_count = 0
    
    for i, (true_pos, pred_pos) in enumerate(zip(true_positions, pred_positions)):
        # Filter out positions outside detector z range
        if true_pos[2] < z_min or true_pos[2] > z_max:
            filtered_count += 1
            continue
        error = np.linalg.norm(pred_pos - true_pos)
        spatial_errors.append(error)
        
        recoil = recoil_types[i] if recoil_types else "Unknown"
        
        # Create hover text
        true_text = (
            f'Sample {i} ({recoil})<br>'
            f'True: ({true_pos[0]:.1f}, {true_pos[1]:.1f}, {true_pos[2]:.1f})<br>'
            f'Error: {error:.1f} mm'
        )
        pred_text = (
            f'Sample {i} ({recoil})<br>'
            f'Pred: ({pred_pos[0]:.1f}, {pred_pos[1]:.1f}, {pred_pos[2]:.1f})<br>'
            f'Error: {error:.1f} mm'
        )
        
        # Separate by recoil type
        if recoil == "NR":
            true_nr_x.append(true_pos[0])
            true_nr_y.append(true_pos[1])
            true_nr_z.append(true_pos[2])
            true_nr_texts.append(true_text)
            
            pred_nr_x.append(pred_pos[0])
            pred_nr_y.append(pred_pos[1])
            pred_nr_z.append(pred_pos[2])
            pred_nr_texts.append(pred_text)
        else:  # ER
            true_er_x.append(true_pos[0])
            true_er_y.append(true_pos[1])
            true_er_z.append(true_pos[2])
            true_er_texts.append(true_text)
            
            pred_er_x.append(pred_pos[0])
            pred_er_y.append(pred_pos[1])
            pred_er_z.append(pred_pos[2])
            pred_er_texts.append(pred_text)
        
        # Add error line connecting true and predicted
        traces.append(go.Scatter3d(
            x=[true_pos[0], pred_pos[0]],
            y=[true_pos[1], pred_pos[1]],
            z=[true_pos[2], pred_pos[2]],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip',
            opacity=0.7
        ))
    
    # Add NR traces (if any)
    if true_nr_x:
        traces.append(go.Scatter3d(
            x=true_nr_x, y=true_nr_y, z=true_nr_z,
            mode='markers',
            marker=dict(size=10, color='green', symbol='square',
                       line=dict(color='darkgreen', width=2)),
            name='True NR',
            text=true_nr_texts,
            hoverinfo='text'
        ))
    
    if pred_nr_x:
        traces.append(go.Scatter3d(
            x=pred_nr_x, y=pred_nr_y, z=pred_nr_z,
            mode='markers',
            marker=dict(size=8, color='green', symbol='x',
                       line=dict(color='darkgreen', width=2)),
            name='Pred NR',
            text=pred_nr_texts,
            hoverinfo='text'
        ))
    
    # Add ER traces (if any)
    if true_er_x:
        traces.append(go.Scatter3d(
            x=true_er_x, y=true_er_y, z=true_er_z,
            mode='markers',
            marker=dict(size=10, color='orange', symbol='square',
                       line=dict(color='darkorange', width=2)),
            name='True ER',
            text=true_er_texts,
            hoverinfo='text'
        ))
    
    if pred_er_x:
        traces.append(go.Scatter3d(
            x=pred_er_x, y=pred_er_y, z=pred_er_z,
            mode='markers',
            marker=dict(size=8, color='orange', symbol='x',
                       line=dict(color='darkorange', width=2)),
            name='Pred ER',
            text=pred_er_texts,
            hoverinfo='text'
        ))
    
    mean_spatial_error = np.mean(spatial_errors) if spatial_errors else 0.0
    
    title_parts = [f'XGBoost Position Predictions ({len(spatial_errors)} samples']
    if filtered_count > 0:
        title_parts[0] += f', {filtered_count} filtered'
    title_parts[0] += ')'
    title_parts.append(f'Mean Spatial Error: {mean_spatial_error:.2f} mm')
    if title_suffix:
        title_parts.append(title_suffix)
    
    title = '<br>'.join([title_parts[0], ' | '.join(title_parts[1:])])
    
    layout = go.Layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        showlegend=True,
        hovermode='closest',
        width=1400,
        height=1000
    )
    
    fig = go.Figure(data=traces, layout=layout)
    
    if save_path is None:
        save_path = Path("xgboost_predictions_3d.html")
    else:
        save_path = save_path.with_suffix('.html')
    
    print("\n" + "="*60)
    print(f"Saving XGBoost 3D predictions to: {save_path}")
    
    fig.write_html(
        str(save_path),
        config={'displayModeBar': True, 'responsive': True},
        include_plotlyjs='cdn'
    )
    
    print(f"✓ Saved! File location: {save_path.absolute()}")
    print("="*60)
    
    return fig


def visualize_xgboost_results(
    model_path: Path,
    data_config: DataConfig = None,
    n_samples: int = 10,
    split: str = "test",
    save_path: Optional[Path] = None,
    random_seed : int = 42,
    
):
    """
    Complete visualization pipeline for XGBoost position predictions.
    
    Args:
        model_path: Path to saved XGBoost model directory
        data_config: Data configuration (uses defaults if None)
        n_samples: Number of samples to visualize
        split: Data split to use
        save_path: Path to save HTML file
        random_seed: Random seed for sample selection
    """
    print("="*60)
    print("XGBoost Position Regressor - Visualization")
    print("="*60)
    
    # Default data config
    if data_config is None:
        data_config = DataConfig(
            access_mode="remote",
            remote_data_path="/ceph/srv/ssjostrom/training_small",
            normalise_positions=True,
            normalise_energy=False,
            normalise_inputs=False,  # XGBoost uses integrated values
        )
    
    # Load detector positions
    position_file = Path("position_MMC_V2.dat")
    if not position_file.exists():
        # Try project root
        position_file = Path(__file__).parent.parent / "position_MMC_V2.dat"
    
    print("\nLoading detector positions...")
    detector_positions = load_detector_positions(position_file)
    
    
    # Load model and make predictions
    true_positions, pred_positions, recoil_types = load_xgboost_model_and_predict(
        model_path=model_path,
        data_config=data_config,
        n_samples=n_samples,
        split=split, 
        random_seed=random_seed
    )
    
    # Create 3D visualization
    if save_path is None:
        save_path = Path("xgboost_predictions_visualization.html")
    
    plot_xgboost_predictions_3d(
        detector_positions=detector_positions,
        true_positions=true_positions,
        pred_positions=pred_positions,
        recoil_types=recoil_types,
        save_path=save_path,
        title_suffix=f"(Split: {split})"
    )
    
    # Print summary statistics
    spatial_errors = [np.linalg.norm(p - t) for p, t in zip(pred_positions, true_positions)]
    
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"  Samples visualized: {len(true_positions)}")
    print(f"  Mean spatial error: {np.mean(spatial_errors):.2f} mm")
    print(f"  Median spatial error: {np.median(spatial_errors):.2f} mm")
    print(f"  Min spatial error: {np.min(spatial_errors):.2f} mm")
    print(f"  Max spatial error: {np.max(spatial_errors):.2f} mm")
    print("="*60)


if __name__ == "__main__":
    # main()  # Transformer model visualization
    
    # ========================================
    # XGBOOST VISUALIZATION CONFIGURATION
    # ========================================
    
    # Easy configuration - change these values:
    N_SAMPLES_TO_VISUALIZE = 10  # Number of test samples to show
    TARGET_ENERGY = 500  # Energy filter (matches training) - set to None for all energies
    MODEL_TIMESTAMP = "20260114_220635"  # Update this to your model's timestamp
    
    # ========================================
    
    
    print("\n" + "="*60)
    print(f"VISUALIZING XGBOOST MODEL PREDICTIONS")
    print(f"  Model: xgboost_regressor_{MODEL_TIMESTAMP}")
    print(f"  Target Energy: {TARGET_ENERGY} eV" if TARGET_ENERGY else "  Target Energy: All energies")
    print(f"  Samples to visualize: {N_SAMPLES_TO_VISUALIZE}")
    print(f"  Split: test set")
    print("="*60 + "\n")
    
    visualize_xgboost_results(
        model_path=Path(f"training_checkpoints/xgboost_regressor_{MODEL_TIMESTAMP}"),
        data_config=DataConfig(
            access_mode="remote",
            remote_data_path="/ceph/srv/ssjostrom/training_small",
            target_energy=TARGET_ENERGY,  # Filter to same energy as training
            normalise_positions=True,  # Match your training config
            normalise_energy=False,
            normalise_inputs=False,
            recoil_types=["NR", "ER"],
            z_include_range=[-1707.0, -1692.0]
             
        ),
        n_samples=N_SAMPLES_TO_VISUALIZE,
        split="test",
        save_path=Path("xgboost_predictions_visualization.html"),
        random_seed=3
    )
    
    # Plot all true positions from test dataset
    # print("\n\n")
    # model_path = Path("training_checkpoints/reconstruction_model_10000.pt")
    # target_energy = 500
    
    # data_config = DataConfig(
    #     access_mode="remote",
    #     remote_data_path="/ceph/srv/ssjostrom/training_small",
    #     normalise_positions=True,
    #     normalise_energy=False,
    #     normalise_inputs=False,
    #     target_energy=target_energy,
    #     normalisation_mode='zscore',
    #     position_norm_cache="/ceph/lwindett/DELight_reconstruction/cache/position_zscore_500eV.pkl",
    # )
    
    # plot_all_true_positions(
    #     model_path=model_path,
    #     data_config=data_config,
    #     split="test",
    #     save_path=Path("all_true_positions_test.html")
    # )