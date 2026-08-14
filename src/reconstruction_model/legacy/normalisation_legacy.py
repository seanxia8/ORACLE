import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import pickle
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TraceNormaliser:
    """
    Compute and apply per-channel normalisation statistics.
    
    Supports zscore mode:
    - 'zscore': (x - mean) / std → mean=0, std=1, unbounded range
    """
    
    def __init__(self, n_channels: int = 56, mode: str = 'zscore'):
        self.n_channels = n_channels
        self.mode = mode  # 'zscore'
        
        # Z-score parameters
        self.channel_means = None
        self.channel_stds = None
        
        self.n_samples_seen = 0
    
    def compute_statistics_from_dataset(
        self,
        dataset,
        cache_path,
        max_samples: int = 10000,
    ):
        """
        Compute normalisation statistics from dataset.
        
        Args:
            dataset: Dataset to compute statistics from
            cache_path: Path to save statistics
            max_samples: Maximum number of samples to use
        """
        logger.info(f"Computing {self.mode} normalisation from {len(dataset)} samples...")
        logger.info(f"Using up to {max_samples} samples for statistics")
        
        if self.mode == 'zscore':
            self._compute_zscore_stats(dataset, max_samples, cache_path)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return self
    
    def _compute_zscore_stats(self, dataset, max_samples, cache_path):
        """Compute mean and std (existing Welford's algorithm)"""
        count = 0
        mean = np.zeros(self.n_channels, dtype=np.float64)
        M2 = np.zeros(self.n_channels, dtype=np.float64)
        
        n_samples_to_use = min(max_samples, len(dataset))
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(dataset)), n_samples_to_use)
        
        for idx in tqdm(sample_indices, desc="Computing zscore stats"):
            try:
                traces, _, _, _ = dataset[idx]
                if isinstance(traces, torch.Tensor):
                    traces = traces.numpy()
                
                trace_channel_means = traces.mean(axis=1)
                
                count += 1
                delta = trace_channel_means - mean
                mean += delta / count
                delta2 = trace_channel_means - mean
                M2 += delta * delta2
                
            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
                continue
        
        if count > 1:
            variance = M2 / (count - 1)
        else:
            variance = np.ones(self.n_channels)
        
        std = np.sqrt(variance)
        std = np.maximum(std, 1e-6)
        
        self.channel_means = mean.astype(np.float32)
        self.channel_stds = std.astype(np.float32)
        self.n_samples_seen = count
        
        logger.info(f"✓ Z-score stats: mean range [{self.channel_means.min():.3f}, {self.channel_means.max():.3f}]")
        logger.info(f"                 std range [{self.channel_stds.min():.3f}, {self.channel_stds.max():.3f}]")
        
        if cache_path:
            self.save(cache_path)
    
    def normalise(self, traces: np.ndarray) -> np.ndarray:
        """Apply normalisation based on mode"""
        if self.mode == 'zscore':
            return self._normalise_zscore(traces)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _normalise_zscore(self, traces: np.ndarray) -> np.ndarray:
        """Z-score: (x - mean) / std"""
        if self.channel_means is None or self.channel_stds is None:
            raise ValueError("Z-score stats not computed")
        
        if traces.ndim == 2:
            mean = self.channel_means[:, np.newaxis]
            std = self.channel_stds[:, np.newaxis]
        elif traces.ndim == 3:
            mean = self.channel_means[np.newaxis, :, np.newaxis]
            std = self.channel_stds[np.newaxis, :, np.newaxis]
        else:
            raise ValueError(f"Expected 2D or 3D, got {traces.shape}")
        
        return (traces - mean) / std
    
    def denormalise(self, traces: np.ndarray) -> np.ndarray:
        """Reverse normalisation"""
        if self.mode == 'zscore':
            return self._denormalise_zscore(traces)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _denormalise_zscore(self, traces: np.ndarray) -> np.ndarray:
        """Reverse: x = x_norm * std + mean"""
        if traces.ndim == 2:
            mean = self.channel_means[:, np.newaxis]
            std = self.channel_stds[:, np.newaxis]
        elif traces.ndim == 3:
            mean = self.channel_means[np.newaxis, :, np.newaxis]
            std = self.channel_stds[np.newaxis, :, np.newaxis]
        else:
            raise ValueError(f"Expected 2D or 3D, got {traces.shape}")
        
        return traces * std + mean
    
    def save(self, path: Path):
        """Save statistics"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'mode': self.mode,
            'n_channels': self.n_channels,
            'n_samples_seen': self.n_samples_seen,
        }
        
        if self.mode == 'zscore':
            stats['channel_means'] = self.channel_means
            stats['channel_stds'] = self.channel_stds
        
        with open(path, 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info(f"Saved {self.mode} normalisation stats to {path}")
    
    def load(self, path: Path):
        """Load statistics"""
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        
        self.mode = stats['mode']
        self.n_channels = stats['n_channels']
        self.n_samples_seen = stats['n_samples_seen']
        
        if self.mode == 'zscore':
            self.channel_means = stats['channel_means']
            self.channel_stds = stats['channel_stds']
        
        logger.info(f"Loaded {self.mode} normalisation from {path}")
        return self

    @classmethod
    def from_cache(cls, path: Path) -> 'TraceNormaliser':
        """Create normaliser and load from cache"""
        normaliser = cls()
        normaliser.load(path)
        return normaliser


class TargetNormaliser:
    """
    normalise spatial coordinates (x, y, z) using z-score.
    
    Computes mean and std for each coordinate independently from training data.
    """
    
    def __init__(self):
        self.means = None  # Shape: (3,) for x, y, z
        self.stds = None   # Shape: (3,) for x, y, z
        self.n_samples_seen = 0
    
    def compute_statistics_from_dataset(
        self,
        dataset,
        max_samples: int = 10000,
        cache_path: Path = None
    ):
        """
        Compute mean and std for x, y, z coordinates using Welford's algorithm.
        """
        logger.info(f"Computing position normalisation from {len(dataset)} samples...")
        logger.info(f"Using up to {max_samples} samples for statistics")
        
        count = 0
        mean = np.zeros(3, dtype=np.float64)  # x, y, z
        M2 = np.zeros(3, dtype=np.float64)
        
        n_samples_to_use = min(max_samples, len(dataset))
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(dataset)), n_samples_to_use)
        
        for idx in tqdm(sample_indices, desc="Computing position stats"):
            try:
                _, spatial_target, _, _ = dataset[idx]
                # Convert to numpy if tensor
                if isinstance(spatial_target, torch.Tensor):
                    position = spatial_target.numpy()
                else:
                    position = np.array(spatial_target)
                
                # Welford's online update
                count += 1
                delta = position - mean
                mean += delta / count
                delta2 = position - mean
                M2 += delta * delta2
                
            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
                continue
        
        if count > 1:
            variance = M2 / (count - 1)
        else:
            variance = np.ones(3)
        
        std = np.sqrt(variance)
        std = np.maximum(std, 1e-6)  
        
        self.means = mean.astype(np.float32)
        self.stds = std.astype(np.float32)
        self.n_samples_seen = count
        
        logger.info(f"✓ Position statistics computed from {count} samples")
        logger.info(f"  X: mean={self.means[0]:7.3f}, std={self.stds[0]:7.3f}")
        logger.info(f"  Y: mean={self.means[1]:7.3f}, std={self.stds[1]:7.3f}")
        logger.info(f"  Z: mean={self.means[2]:7.3f}, std={self.stds[2]:7.3f}")
        
        if cache_path:
            self.save(cache_path)
        
        return self
    
    
    def normalise(self, positions: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalisation to positions.
        
        Args:
            positions: (3,) or (batch, 3) array of (x, y, z)
        
        Returns:
            normalised positions with mean≈0, std≈1
        """
        if self.means is None or self.stds is None:
            raise ValueError("Statistics not computed. Call compute_statistics_from_dataset() first.")
        
        return (positions - self.means) / self.stds
    
    
    def denormalise(self, positions: np.ndarray) -> np.ndarray:
        """Reverse normalisation: positions = normalised * std + mean"""
        if self.means is None or self.stds is None:
            raise ValueError("Statistics not computed.")
        
        return positions * self.stds + self.means
    
    
    def save(self, path: Path):
        """Save statistics to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'means': self.means,
            'stds': self.stds,
            'n_samples_seen': self.n_samples_seen
        }
        
        with open(path, 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info(f"Saved position normalisation stats to {path}")
    
    
    def load(self, path: Path):
        """Load statistics from disk"""
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        
        self.means = stats['means']
        self.stds = stats['stds']
        self.n_samples_seen = stats['n_samples_seen']
        
        logger.info(f"Loaded position normalisation from {path}")
        logger.info(f"  Based on {self.n_samples_seen} samples")
        return self
    
    @classmethod
    def from_cache(cls, path: Path) -> 'TargetNormaliser':
        """Create normaliser and load from cache"""
        normaliser = cls()
        normaliser.load(path)
        return normaliser


def compute_or_load_target_normaliser(
    dataset,
    cache_path: Path,
    force_recompute: bool = False,
    max_samples: int = 10000
) -> TargetNormaliser:
    """
    Smart wrapper: load cached position stats OR compute from dataset.
    """
    cache_path = Path(cache_path)
    
    if cache_path.exists() and not force_recompute:
        logger.info(f"Found cached position statistics: {cache_path}")
        try:
            return TargetNormaliser.from_cache(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, recomputing...")
    
    logger.info("Computing new position normalisation statistics...")
    normaliser = TargetNormaliser()
    normaliser.compute_statistics_from_dataset(
        dataset,
        max_samples=max_samples,
        cache_path=cache_path
    )
    
    return normaliser


def compute_or_load_normaliser_from_dataset(
    dataset,
    cache_path: Path,
    force_recompute: bool = False,
    max_samples: int = 10000,
    mode: str = 'zscore'  
) -> TraceNormaliser:
    """Compute or load normaliser"""
    cache_path = Path(cache_path)
    
    if cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached normalisation...")
        return TraceNormaliser.from_cache(cache_path)
    
    logger.info(f"Computing new {mode} normalisation...")
    normaliser = TraceNormaliser(mode=mode) 
    normaliser.compute_statistics_from_dataset(
        dataset,
        max_samples=max_samples,
        cache_path=cache_path
    )
    
    return normaliser


def compute_energy_stats_from_dataset(dataset, cache_path: Path = None, force_recompute: bool = False) -> tuple[float, float]:
    """
    Compute mean and std from unique energy values in dataset.
    
    Since energies are discrete (e.g., [10, 20, 50, 100, 200, 500]),
    we just extract unique values and compute stats directly.
    
    Args:
        dataset: Dataset to extract energies from
        cache_path: Path to cache energy stats
        force_recompute: Force recomputation even if cache exists
    
    Returns:
        (mean, std) tuple
    """
    cache_path = Path(cache_path) if cache_path else None
    
    # Try to load from cache first
    if cache_path and cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached energy stats from {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                stats = pickle.load(f)
            mean = stats['mean']
            std = stats['std']
            logger.info(f"✓ Loaded energy stats: mean={mean:.4f}, std={std:.4f}")
            return mean, std
        except Exception as e:
            logger.warning(f"Failed to load energy cache: {e}, recomputing...")
    
    logger.info("Extracting unique energy values from training data...")
    
    # Collect unique energies from sample_index
    unique_energies = set()
    for sample_info in dataset.sample_index:
        unique_energies.add(sample_info['energy'])
    
    unique_energies = sorted(unique_energies)
    logger.info(f"Found unique energy values: {unique_energies} eV")
    
    energy_array = np.array(unique_energies)
    
    # Compute mean and std
    mean = float(np.mean(energy_array))
    std = float(np.std(energy_array))
    
    # Ensure std is not zero
    if std < 1e-6:
        logger.warning("Energy std is near zero, setting to 1.0")
        std = 1.0
    
    logger.info(f"✓ Energy normalization stats: mean={mean:.4f}, std={std:.4f}")
    
    # Save to cache if path provided
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {'mean': mean, 'std': std}
        with open(cache_path, 'wb') as f:
            pickle.dump(stats, f)
        logger.info(f"Saved energy stats to {cache_path}")
    
    return mean, std