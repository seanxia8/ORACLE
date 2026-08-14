"""
Compute position normalization statistics for visualization.

This script computes z-score normalization statistics for positions only,
using 500 eV data from the small dataset.
"""
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruction_model.dataset_temp import DataConfig, create_dataloaders

def main():
    print("="*60)
    print("Computing Position Normalization Statistics")
    print("="*60)
    
    # Configuration
    target_energy = 500
    cache_dir = Path("/ceph/lwindett/DELight_reconstruction/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    position_cache_path = cache_dir / "position_zscore_500eV.pkl"
    
    print(f"\nConfiguration:")
    print(f"  Target energy: {target_energy} eV")
    print(f"  Dataset: /ceph/srv/ssjostrom/training_small")
    print(f"  Output cache: {position_cache_path}")
    print(f"  Normalization mode: zscore")
    print()
    
    # Create data config - only normalize positions
    data_config = DataConfig(
        access_mode="remote",
        remote_data_path="/ceph/srv/ssjostrom/training_small",
        recoil_types=["NR", "ER"],  # Need to specify recoil types
        target_energy=target_energy,
        normalise_positions=True,  # Only this
        normalise_energy=False,     # Not energy
        normalise_inputs=False,     # Not inputs
        normalisation_mode='zscore',  # Use z-score normalization
        position_norm_cache=str(position_cache_path),
    )
    
    print("Creating dataloaders to compute position statistics...")
    print("This will compute z-score (mean, std) from the dataset.\n")
    
    # Create dataloaders - this will compute and cache the stats
    dataloaders = create_dataloaders(
        data_config=data_config,
        batch_size=32,
        num_workers=0,
        max_samples_for_norm=5000,  # Use 5000 samples for good statistics
        precomputed_trace=False,     # Don't need trace normalization
        precomputed_positions=False, # Compute new position stats
        precomputed_energy=False,    # Don't need energy normalization
    )
    
    print("\n" + "="*60)
    print("✓ Position statistics computed and cached!")
    print(f"✓ Cache location: {position_cache_path}")
    print("="*60)
    
    # Verify the cache was created
    if position_cache_path.exists():
        print(f"\n✓ Verified: Cache file exists ({position_cache_path.stat().st_size} bytes)")
    else:
        print(f"\n✗ Warning: Cache file not found at {position_cache_path}")
    
    print("\nYou can now use this cache in your visualization script:")
    print(f"  position_norm_cache=\"{position_cache_path}\"")


if __name__ == "__main__":
    main()
