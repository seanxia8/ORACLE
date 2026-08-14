import io
import os
import re
import subprocess
import logging
from collections import OrderedDict
from typing import Tuple, Optional, Dict, Literal
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import h5py
import zstandard as zstd
import pandas as pd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_CONFIG_DIR = Path(__file__).parent
_PROJECT_ROOT = _CONFIG_DIR.parent

@dataclass
class DataConfig:
    """Configuration for dataset parameters"""

    access_mode: Literal["remote", "local"] = "local"
    data_format: Literal["zst", "h5_batch"] = "zst"
    # SSH/Remote access
    ssh_host: Optional[str] = None
    remote_data_path: str = "/ceph/dwong/work"
    
    # Local access for GPU nodes
    local_data_path: str = _PROJECT_ROOT / "training_data"
    local_cache_path: str = "cache"  # Only for small metadata files now

    # Data structure paths
    train_path: str = "train"  # Contains ER/ and NR/ subdirs
    test_path: str = "threshold"  # need to change this to "test" later

    # Data parameters
    max_seq_len: int = 65536
    recoil_types: list = None  # ["ER", "NR"] for training, None for test
    energies: Optional[list] = None
    max_h5_files_per_energy_recoil: Optional[int] = None
    expected_h5_events_per_file: Optional[int] = None
    max_open_h5_files: int = 32
    train_split: float = 0.8  # Split within training data for train/val
    val_split: float = 0.2

    # Testing parameters  
    dummy_mode: bool = False

    def __post_init__(self):
        if self.recoil_types is None:
            self.recoil_types = ["ER", "NR"]

    def get_base_data_path(self) -> str:
        """Get the appropriate base data path based on access mode"""
        if self.access_mode == "remote":
            return self.remote_data_path
        else:
            return self.local_data_path


def read_meta_h5(meta_path: Path):
    """
    Returns (attrs: dict, df: pandas.DataFrame) for meta file.
    Expects dataset 'events' and attrs: n_channels, trace_samples, trace_dtype.
    """
    meta_path = Path(meta_path)
    with h5py.File(meta_path, "r") as f:
        attrs = {
            k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()
        }
        data = f["events"][:]  # structured array
    df = pd.DataFrame(
        {
            "x": data["x"],
            "y": data["y"],
            "z": data["z"],
            "energy": data["energy"],
            "type_recoil": [
                s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                for s in data["type_recoil"]
            ],
            "no_noise": data["no_noise"],
            "quantize": data["quantize"],
        }
    )
    return attrs, df


# ---------- vectorized unshuffle for a whole batch ----------
def _unshuffle_batch(
    block: bytes,
    batch_events: int,
    n_channels: int,
    trace_samples: int,
    dtype=np.float16,
) -> np.ndarray:
    """
    Inverse of the byte-shuffle used when writing traces.
    Input 'block' holds concatenated shuffled traces for 'batch_events'.
    Returns array with shape (batch_events, n_channels, trace_samples).
    """
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize
    num_elements = n_channels * trace_samples

    u8 = np.frombuffer(block, dtype=np.uint8)
    expected = batch_events * itemsize * num_elements
    if u8.size != expected:
        raise ValueError(
            f"Unexpected batch size: got {u8.size} bytes, expected {expected}"
        )
    # [B, itemsize, N] -> [B, N, itemsize] -> flatten last 2 dims, then view
    u8 = (
        u8.reshape(batch_events, itemsize, num_elements)
        .swapaxes(1, 2)
        .reshape(batch_events, num_elements * itemsize)
    )
    arr = u8.view(dtype)  # (B, N)
    return arr.reshape(batch_events, n_channels, trace_samples)


# ---------- single-file batched iterator ----------
def iter_traces_zst_batched_merged(
    traces_path: Path,
    n_events: int,
    n_channels: int,
    trace_samples: int,
    batch_size: int = 1000,
    dtype=np.float16,
    max_events: Optional[int] = None,
):
    """
    Iterate over a merged .zst file in *batches*, yielding arrays of shape
    (B, n_channels, trace_samples) where B <= batch_size.
    - Works on a *single* large file written by concatenating event frames.
    - Uses zstd streaming + vectorized unshuffle for speed.
    - If max_events is set, stops after ~max_events (last batch may overrun slightly).
    """
    dtype = np.dtype(dtype)
    per_event_bytes = int(n_channels * trace_samples * dtype.itemsize)
    to_read = n_events if max_events is None else min(max_events, n_events)

    dctx = zstd.ZstdDecompressor()
    with open(traces_path, "rb") as fin, dctx.stream_reader(fin) as reader:
        buf = io.BufferedReader(reader)
        remaining = to_read
        while remaining > 0:
            bsz = int(min(batch_size, remaining))
            need = per_event_bytes * bsz
            got, chunks = 0, []
            while got < need:
                chunk = buf.read(need - got)
                if not chunk:
                    raise EOFError(
                        f"Unexpected end of stream: need {need} bytes, got {got} bytes"
                    )
                chunks.append(chunk)
                got += len(chunk)
            block = b"".join(chunks)
            yield _unshuffle_batch(block, bsz, n_channels, trace_samples, dtype=dtype)
            remaining -= bsz


# ---------- convenience wrapper ----------
def open_merged_dataset(base_dir: Path, energy: int):
    """
    Reads meta + returns an iterator factory over the merged traces file.
    Usage:
        attrs, meta_df, get_iter = open_merged_dataset(Path(BASE), 100)
        for batch in get_iter(batch_size=1000, dtype=np.float16, max_events=5000):
            ...
    """
    base = Path(base_dir)
    meta_path = base / f"meta_energy_{energy}.h5"
    traces_path = base / f"traces_energy_{energy}.zst"
    attrs, meta_df = read_meta_h5(meta_path)
    n_events = len(meta_df)
    n_channels = int(attrs["n_channels"])
    trace_samples = int(attrs["trace_samples"])
    trace_dtype = np.dtype(attrs.get("trace_dtype", np.float16))

    def get_iter(batch_size=1000, dtype=trace_dtype, max_events=None):
        return iter_traces_zst_batched_merged(
            traces_path,
            n_events,
            n_channels,
            trace_samples,
            batch_size=batch_size,
            dtype=dtype,
            max_events=max_events,
        )

    return attrs, meta_df, get_iter


class RemoteDataManager:
    """Manages remote data access and caching for datasets."""

    def __init__(self, config: DataConfig):
        self.config = config

        # Set up paths based on access mode
        if config.access_mode == "local":
            self.data_path = Path(config.local_data_path)
            self.local_cache = Path(config.local_cache_path)
        else:  # remote mode
            self.data_path = Path(config.remote_data_path)
            self.local_cache = Path(config.local_cache_path)

        self.local_cache.mkdir(parents=True, exist_ok=True)

        # Create HDF5 cache subdirectory
        self.hdf5_cache_dir = self.local_cache / "hdf5_cache"
        self.hdf5_cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for opened datasets to avoid reopening files
        self.dataset_cache = {}
        # Cache for HDF5 file handles
        self.hdf5_handles = {}

    def _get_hdf5_path(self, energy: int, recoil_type: str, base_path: str):
        """Get the HDF5 cache file path for a given energy/recoil combination"""
        # Create safe filename from base_path and recoil_type
        path_hash = abs(hash(base_path)) % 10000
        recoil_safe = recoil_type.replace('/', '_').replace('\\', '_')
        filename = f"energy_{energy}_{recoil_safe}_{path_hash}.h5"
        return self.hdf5_cache_dir / filename

    def _convert_zst_to_hdf5(self, energy: int, recoil_type: str, base_path: str):
        """Convert .zst file to HDF5 for fast random access"""
        hdf5_path = self._get_hdf5_path(energy, recoil_type, base_path)
        
        # Check if already converted
        if hdf5_path.exists():
            print(f"Using existing HDF5: {hdf5_path.name}")
            return hdf5_path
        
        print(f"Converting to HDF5: {energy}keV {recoil_type}...")
        start_time = time.time()
        
        try:
            # Load original data using existing functions
            attrs, meta_df, get_iter = open_merged_dataset(Path(base_path), energy)
            
            # Filter metadata by recoil type if needed
            original_size = len(meta_df)
            if recoil_type != "mixed":
                meta_df = meta_df[meta_df["type_recoil"] == recoil_type].copy()
                meta_df.reset_index(drop=True, inplace=True)
            
            n_events = len(meta_df)
            n_channels = int(attrs["n_channels"])
            trace_samples = int(attrs["trace_samples"])
            
            print(f"  Converting {n_events} events (filtered from {original_size})")
            print(f"  Channels: {n_channels}, Samples: {trace_samples}")
            
            # Create temporary file to avoid corruption if interrupted
            temp_path = hdf5_path.with_suffix('.h5.tmp')
            
            with h5py.File(temp_path, 'w') as h5f:
                # Store attributes
                for key, value in attrs.items():
                    h5f.attrs[key] = value
                h5f.attrs['filtered_recoil_type'] = recoil_type
                
                # Create dataset for traces - chunked for efficient row access
                chunk_size = min(100, n_events) 
                traces_dataset = h5f.create_dataset(
                    'traces',
                    shape=(n_events, n_channels, trace_samples),
                    dtype=np.float32,
                    chunks=(chunk_size, n_channels, trace_samples),
                    compression='gzip',
                    compression_opts=1,  # Light compression for speed
                    shuffle=True  # Improve compression
                )
                
                # Store metadata as separate datasets
                meta_group = h5f.create_group('metadata')
                for col in meta_df.columns:
                    if col == 'type_recoil':
                        # Handle string data
                        string_data = [s.encode('utf-8') for s in meta_df[col].values]
                        meta_group.create_dataset(col, data=string_data)
                    else:
                        meta_group.create_dataset(col, data=meta_df[col].values)
                
                # Load and store trace data in batches
                batch_size = 1000
                iterator = get_iter(batch_size=batch_size)
                current_idx = 0

                for batch_traces in iterator:
                    batch_size_actual = batch_traces.shape[0]
                    
                    if current_idx + batch_size_actual <= n_events:
                        traces_dataset[current_idx:current_idx + batch_size_actual] = batch_traces.astype(np.float32)
                    
                    current_idx += batch_size_actual
                    
                    if current_idx % 5000 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Processed {current_idx}/{len(meta_df)} events ({elapsed:.1f}s)")
                    
                    if current_idx >= n_events:
                        break
            
            # Move temp file to final location
            temp_path.rename(hdf5_path)
            
            elapsed = time.time() - start_time
            file_size_mb = hdf5_path.stat().st_size / (1024 * 1024)
            print(f"✓ Converted to HDF5 in {elapsed:.1f}s ({file_size_mb:.1f} MB)")
            return hdf5_path
            
        except Exception as e:
            # Clean up temp files
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            if hdf5_path.exists():
                hdf5_path.unlink()
            logger.error(f"Failed to convert {energy}keV {recoil_type}: {e}")
            raise

    def get_hdf5_dataset(self, energy: int, recoil_type: str, base_path: str):
        """Get HDF5 dataset handle for fast indexing"""
        cache_key = f"{base_path}_{energy}_{recoil_type}"
        
        if cache_key not in self.hdf5_handles:
            # Convert to HDF5 if needed
            hdf5_path = self._convert_zst_to_hdf5(energy, recoil_type, base_path)
            
            # Open HDF5 file
            h5f = h5py.File(hdf5_path, 'r')
            
            # Load metadata into pandas DataFrame
            meta_dict = {}
            for col in h5f['metadata'].keys():
                data = h5f['metadata'][col][:]
                if col == 'type_recoil':
                    # Decode string data
                    data = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in data]
                meta_dict[col] = data
            
            meta_df = pd.DataFrame(meta_dict)
            
            self.hdf5_handles[cache_key] = {
                'h5_file': h5f,
                'traces': h5f['traces'],  # Direct access to traces dataset
                'metadata': meta_df,
                'attrs': dict(h5f.attrs),
                'n_events': len(meta_df)
            }
        
        return self.hdf5_handles[cache_key]

    def get_single_sample(self, energy: int, recoil_type: str, base_path: str, event_idx: int):
        """Get a single sample by index - very fast random access!"""
        try:
            dataset_info = self.get_hdf5_dataset(energy, recoil_type, base_path)
            
            if event_idx >= dataset_info['n_events']:
                return None
            
            # Direct indexing - this is extremely fast!
            traces = dataset_info['traces'][event_idx]  # Shape: (n_channels, trace_samples)
            metadata_row = dataset_info['metadata'].iloc[event_idx]
            
            return traces, metadata_row
            
        except Exception as e:
            logger.error(f"Failed to get sample {event_idx} for {energy}keV {recoil_type}: {e}")
            return None

    def __del__(self):
        """Clean up HDF5 files when manager is destroyed"""
        for handle_info in self.hdf5_handles.values():
            if 'h5_file' in handle_info:
                try:
                    handle_info['h5_file'].close()
                except:
                    pass

    def _run_ssh_command(self, command: str) -> str:
        """Execute command via SSH"""
        if not self.config.ssh_host:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        else:
            ssh_command = f"ssh {self.config.ssh_host} '{command}'"
            result = subprocess.run(
                ssh_command, shell=True, capture_output=True, text=True
            )

        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {command}\nError: {result.stderr}")
        return result.stdout.strip()

    def list_energy_files(self, split: str) -> Dict[str, list]:
        """List available energy files and metadata - avoiding duplicate counting"""
        files = {"energy_levels": [], "metadata": []}

        base_data_path = self.config.get_base_data_path() 
        if split in ["train", "val"]:
            if self.config.access_mode == "local":
                # Local file system access
                base_path = Path(base_data_path) / self.config.train_path
                
                for recoil_type in self.config.recoil_types:
                    recoil_path = base_path / recoil_type
                    
                    if not recoil_path.exists():
                        logger.warning(
                            f"Local path does not exist: {recoil_path}\nCurrent file path: {Path(__file__).resolve()}"
                            f"\nBase path: {base_path}\n"
                            )
                        continue

                    # Find all traces_energy_*.zst files locally
                    for traces_file in recoil_path.glob("traces_energy_*.zst"):
                        filename = traces_file.name
                        if filename.startswith("traces_energy_") and filename.endswith(".zst"):
                            energy_str = filename[len("traces_energy_"):-len(".zst")]
                            try:
                                energy = int(energy_str)
                                
                                # Check corresponding metadata file exists
                                meta_file = recoil_path / f"meta_energy_{energy}.h5"
                                
                                if meta_file.exists():
                                    files["energy_levels"].append({
                                        "energy": energy,
                                        "recoil_type": recoil_type,
                                        "base_path": str(recoil_path),
                                    })
                                    
                                    if str(meta_file) not in files["metadata"]:
                                        files["metadata"].append(str(meta_file))
                                else:
                                    logger.warning(f"Missing metadata file: {meta_file}")
                                    
                            except ValueError:
                                logger.warning(f"Could not parse energy from filename: {filename}")
                                continue

            else:
                # Remote SSH access (your existing logic)
                base_path = f"{base_data_path}/{self.config.train_path}"

                # First, discover all unique energy levels across all recoil types
                all_energies = set()

                for recoil_type in self.config.recoil_types:
                    recoil_path = f"{base_path}/{recoil_type}"

                    # Look for traces_energy_*.zst files
                    command = f"find {recoil_path} -name 'traces_energy_*.zst' 2>/dev/null || true"
                    zst_files = self._run_command(command).split("\n")

                    for file_path in zst_files:
                        if file_path.strip():
                            filename = Path(file_path).name
                            if filename.startswith("traces_energy_") and filename.endswith(".zst"):
                                energy_str = filename[len("traces_energy_") : -len(".zst")]
                                try:
                                    energy = int(energy_str)
                                    all_energies.add(energy)
                                except ValueError:
                                    continue

                # Now create entries for each unique (energy, recoil_type) combination
                for energy in sorted(all_energies):
                    for recoil_type in self.config.recoil_types:
                        recoil_path = f"{base_path}/{recoil_type}"

                        # Check if this specific energy file exists for this recoil type
                        traces_file = f"{recoil_path}/traces_energy_{energy}.zst"
                        meta_file = f"{recoil_path}/meta_energy_{energy}.h5"

                        # Verify both files exist
                        # check_command = f"test -f '{traces_file}' && test -f '{meta_file}' && echo 'exists' || echo 'missing'"
                        result = self._run_command(check_command).strip()

                        if result == "exists":
                            files["energy_levels"].append({
                                "energy": energy,
                                "recoil_type": recoil_type,
                                "base_path": recoil_path,
                            })

                            if meta_file not in files["metadata"]:
                                files["metadata"].append(meta_file)

        elif split == "test":
            if self.config.access_mode == "local":
                # Local test data access
                base_path = Path(base_data_path) / self.config.test_path
                
                if base_path.exists():
                    for traces_file in base_path.glob("traces_energy_*.zst"):
                        filename = traces_file.name
                        if filename.startswith("traces_energy_") and filename.endswith(".zst"):
                            energy_str = filename[len("traces_energy_"):-len(".zst")]
                            try:
                                energy = int(energy_str)
                                
                                meta_file = base_path / f"meta_energy_{energy}.h5"
                                
                                if meta_file.exists():
                                    files["energy_levels"].append({
                                        "energy": energy,
                                        "recoil_type": "mixed",
                                        "base_path": str(base_path),
                                    })
                                    
                                    if str(meta_file) not in files["metadata"]:
                                        files["metadata"].append(str(meta_file))
                            except ValueError:
                                continue
                else:
                    logger.warning(f"Test path does not exist: {base_path}")

            else:
                # Remote test data access
                base_path = f"{base_data_path}/{self.config.test_path}"

                command = f"find {base_path} -name 'traces_energy_*.zst' 2>/dev/null || true"
                zst_files = self._run_command(command).split("\n")

                discovered_energies = set()

                for file_path in zst_files:
                    if file_path.strip():
                        filename = Path(file_path).name
                        if filename.startswith("traces_energy_") and filename.endswith(".zst"):
                            energy_str = filename[len("traces_energy_") : -len(".zst")]
                            try:
                                energy = int(energy_str)
                                if energy not in discovered_energies:
                                    discovered_energies.add(energy)
                                    files["energy_levels"].append({
                                        "energy": energy,
                                        "recoil_type": "mixed",
                                        "base_path": base_path,
                                    })
                            except ValueError:
                                continue

                # Look for metadata files
                command = f"find {base_path} -name 'meta_energy_*.h5' 2>/dev/null || true"
                metadata_files = self._run_command(command).split("\n")
                files["metadata"].extend([f for f in metadata_files if f.strip()])

        logger.info(f"Found {len(files['energy_levels'])} energy levels for {split} (access_mode: {self.config.access_mode})")
        return files

    def get_dataset_iterator(self, energy: int, recoil_type: str, base_path: str):
        """Get iterator for a specific energy level and recoil type"""
        cache_key = f"{base_path}_{energy}"

        if cache_key not in self.dataset_cache:
            try:
                attrs, meta_df, get_iter = open_merged_dataset(Path(base_path), energy)

                # Filter metadata by recoil type if not "mixed"
                if recoil_type != "mixed":
                    meta_df = meta_df[meta_df["type_recoil"] == recoil_type].copy()

                self.dataset_cache[cache_key] = {
                    "attrs": attrs,
                    "metadata": meta_df,
                    "get_iter": get_iter,
                }

            except Exception as e:
                logger.error(
                    f"Failed to load dataset for energy {energy}, recoil {recoil_type}: {e}"
                )
                raise

        return self.dataset_cache[cache_key]

    


class ParticleReconstructionDataset(Dataset):
    """Dataset for efficient loading"""

    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        transform: Optional[callable] = None,
    ):
        self.config = config
        self.split = split
        self.transform = transform

        if not config.dummy_mode:
            # Setup remote data manager
            self.data_manager = RemoteDataManager(config)

            # Load available energy levels and create sample index
            self.energy_levels, self.metadata_files = self._load_file_structure()
            self.sample_index = []
            self._create_sample_index()
            if split == "test":
                self.split_samples = self.sample_index
            else:
                self.split_samples = self._create_train_val_splits()
                self.verify_split_distribution()

        else:
            self._create_dummy_sample_index()

        logger.info(f"Initialized {split} dataset with {len(self)} samples")

    def _load_file_structure(self):
        """Discover available energy files"""
        file_info = self.data_manager.list_energy_files(self.split)
        energy_levels = file_info["energy_levels"]
        metadata_files = file_info["metadata"]

        if not energy_levels:
            raise ValueError(f"No energy files found for split: {self.split}")

        return energy_levels, metadata_files

    def _create_dummy_sample_index(self):
        """Create fake sample index for testing"""
        # Create realistic sample distribution
        energies = [50, 100, 200, 500]  # Typical energies
        recoil_types = ["ER", "NR"]
        
        self.sample_index = []
        for energy in energies:
            for recoil_type in recoil_types:
                # Create 100 fake samples per energy/recoil combo
                for event_idx in range(100):
                    self.sample_index.append({
                        'energy': energy,
                        'recoil_type': recoil_type,
                        'base_path': f'/fake/path/{energy}keV',
                        'event_idx': event_idx
                    })
        
        # Create train/val splits if needed
        if self.split != "test":
            np.random.seed(0)
            indices = np.random.permutation(len(self.sample_index))
            train_end = int(self.config.train_split * len(indices))
            
            if self.split == "train":
                selected_indices = indices[:train_end]
            else:  # val
                selected_indices = indices[train_end:]
            
            self.split_samples = [self.sample_index[i] for i in selected_indices]

    def _create_sample_index(self):
        """Create an index mapping sample indices to (energy, recoil_type, event_idx)"""
        for energy_info in self.energy_levels:
            energy = energy_info["energy"]
            recoil_type = energy_info["recoil_type"]
            base_path = energy_info["base_path"]

            try:
                # Get dataset info to find number of events
                dataset_info = self.data_manager.get_dataset_iterator(
                    energy, recoil_type, base_path
                )
                n_events = len(dataset_info["metadata"])

                # Add entries for each event in this energy level
                for event_idx in range(n_events):
                    self.sample_index.append(
                        {
                            "energy": energy,
                            "recoil_type": recoil_type,
                            "base_path": base_path,
                            "event_idx": event_idx,
                        }
                    )

            except Exception as e:
                logger.warning(
                    f"Could not load energy {energy}, recoil {recoil_type}: {e}"
                )
                continue

    def _create_train_val_splits(self):
        """Create train/val splits ensuring both splits have all energy levels"""
        # Group by (energy, recoil_type) to split each energy level separately
        energy_recoil_groups = {}
        for sample_info in self.sample_index:
            energy = sample_info["energy"]
            recoil_type = sample_info["recoil_type"]
            key = (energy, recoil_type)

            if key not in energy_recoil_groups:
                energy_recoil_groups[key] = []
            energy_recoil_groups[key].append(sample_info)

        split_samples = []

        # Split EACH (energy, recoil_type) combination separately
        np.random.seed(0)
        for (energy, recoil_type), samples in energy_recoil_groups.items():
            n_samples = len(samples)
            indices = np.random.permutation(n_samples)

            train_end = int(self.config.train_split * n_samples)

            if self.split == "train":
                selected_indices = indices[:train_end]
            else:  # val
                selected_indices = indices[train_end:]

            # Add the selected samples for this energy/recoil combination
            for idx in selected_indices:
                split_samples.append(samples[idx])

            print(
                f"Energy {energy}, Recoil {recoil_type}: "
                f"{len(selected_indices)} samples for {self.split} "
                f"(out of {n_samples} total)"
            )

        if not split_samples:
            raise ValueError(f"No samples found for split: {self.split}")

        # Print summary of energy distribution
        energy_counts = {}
        for sample in split_samples:
            energy = sample["energy"]
            energy_counts[energy] = energy_counts.get(energy, 0) + 1

        print(f"{self.split} split energy distribution: {energy_counts}")
        return split_samples

    def verify_split_distribution(self):
        """Verify that train/val splits have good energy distribution"""
        energy_distribution = {}
        recoil_distribution = {}

        for sample in self.split_samples:
            energy = sample["energy"]
            recoil = sample["recoil_type"]

            energy_distribution[energy] = energy_distribution.get(energy, 0) + 1
            recoil_distribution[recoil] = recoil_distribution.get(recoil, 0) + 1

        print(f"\n{self.split.upper()} SPLIT VERIFICATION:")
        print(f"Energy levels: {sorted(energy_distribution.keys())}")
        print(f"Energy distribution: {energy_distribution}")
        print(f"Recoil distribution: {recoil_distribution}")

        return energy_distribution, recoil_distribution

    def __len__(self) -> int:
        if hasattr(self, "split_samples"):
            return len(self.split_samples)
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Fast single sample loading using HDF5 direct indexing"""
        if hasattr(self, "split_samples"):
            sample_info = self.split_samples[idx]
        else:
            sample_info = self.sample_index[idx]
        
        if self.config.dummy_mode: 
            # Generate realistic dummy data
            np.random.seed(sample_info['energy'] * 1000 + sample_info['event_idx'])
            
            traces = np.random.randn(56, self.config.max_seq_len).astype(np.float32) * 0.1
            
            signal_amplitude = sample_info['energy'] / 1000.0
            traces[:, :100] += signal_amplitude * np.random.exponential(0.5, (56, 100))
            
            input_tensor = torch.from_numpy(traces).float()
            
            
            x = np.random.uniform(-50, 50)
            y = np.random.uniform(-50, 50) 
            z = np.random.uniform(0, 100)
            spatial_target = torch.tensor([x, y, z]).float()
            
            energy_target = torch.tensor(float(sample_info['energy'])).float()
            recoil_type = sample_info['recoil_type']
            
            return input_tensor, spatial_target, energy_target, recoil_type

        else: 
            try:
                # Load single sample directly by index - this is extremely fast!
                sample_data = self.data_manager.get_single_sample(
                    energy=sample_info["energy"],
                    recoil_type=sample_info["recoil_type"],
                    base_path=sample_info["base_path"],
                    event_idx=sample_info["event_idx"]
                )

                if sample_data is None:
                    return self._get_dummy_sample()

                traces, metadata_row = sample_data

                # Convert to tensors
                input_tensor = torch.from_numpy(traces).float()

                # Get spatial coordinates and energy from metadata
                spatial_target = torch.tensor([
                    metadata_row["x"], 
                    metadata_row["y"], 
                    metadata_row["z"]
                ]).float()

                energy_target = torch.tensor(metadata_row["energy"]).float()
                recoil_type = metadata_row["type_recoil"]

                # Handle sequence length
                if input_tensor.size(-1) > self.config.max_seq_len:
                    input_tensor = input_tensor[..., :self.config.max_seq_len]
                elif input_tensor.size(-1) < self.config.max_seq_len:
                    pad_size = self.config.max_seq_len - input_tensor.size(-1)
                    input_tensor = F.pad(input_tensor, (0, pad_size))

                # Apply transforms
                if self.transform:
                    input_tensor = self.transform(input_tensor)

                return input_tensor, spatial_target, energy_target, recoil_type

            except Exception as e:
                logger.error(f"Failed to load sample {idx}: {e}")
                return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """Return dummy sample in case of loading errors"""
        input_tensor = torch.zeros(
            56, self.config.max_seq_len  
        )
        spatial_target = torch.zeros(3)
        energy_target = torch.tensor(100.0)  
        return input_tensor, spatial_target, energy_target, "ER"

    def debug_channel_counts(self):
        """Debug function to check channel counts across energy levels"""
        print("=" * 50)
        print("Checking channel counts across energy levels...")
        
        channel_counts = {}
        
        for energy_info in self.energy_levels:
            try:
                # Get dataset info to check attributes
                dataset_info = self.data_manager.get_dataset_iterator(
                    energy_info["energy"], 
                    energy_info["recoil_type"], 
                    energy_info["base_path"]
                )
                
                attrs = dataset_info['attrs']
                n_channels = int(attrs["n_channels"])
                trace_samples = int(attrs["trace_samples"])
                energy = energy_info["energy"]
                recoil = energy_info["recoil_type"]
                
                key = f"{energy}keV_{recoil}"
                channel_counts[key] = {
                    'channels': n_channels,
                    'samples': trace_samples
                }
                
            except Exception as e:
                print(f"Error checking {energy_info}: {e}")
        
        print("Channel/sample counts by energy/recoil:")
        for key, info in channel_counts.items():
            print(f"  {key}: {info['channels']} channels, {info['samples']} trace samples")
        
        unique_channels = set(info['channels'] for info in channel_counts.values())
        unique_samples = set(info['samples'] for info in channel_counts.values())
        
        print(f"\nUnique channel counts: {sorted(unique_channels)}")
        print(f"Unique trace sample counts: {sorted(unique_samples)}")
        
        # Check if all are the same
        if len(unique_channels) == 1:
            print("✓ All energy levels have the same number of channels")
            print("  → No collate function padding needed for channels")
        else:
            print("⚠️  Different channel counts detected!")
            print("  → Need collate function with channel padding")
        
        if len(unique_samples) == 1:
            print("✓ All energy levels have the same trace length")
        else:
            print("⚠️  Different trace lengths detected!")
            print("  → Sequence length handling needed")
        
        return channel_counts


class H5BatchParticleReconstructionDataset(Dataset):
    """Dataset for H5 batch files like ER_traces_energy_100_batch_0000.h5."""

    _FILENAME_RE = re.compile(
        r"(?P<recoil>ER|NR)_traces_energy_(?P<energy>\d+)_batch_(?P<batch>\d+)\.h5$"
    )

    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        transform: Optional[callable] = None,
    ):
        self.config = config
        self.split = split
        self.transform = transform
        self.base_path = self._resolve_base_path(Path(config.local_data_path))
        self.sample_index = []
        self._file_handles = OrderedDict()

        self._create_sample_index()
        if split == "test":
            self.split_samples = self.sample_index
        else:
            self.split_samples = self._create_train_val_splits()
            self.verify_split_distribution()

        logger.info(
            "Initialized %s H5 batch dataset with %d samples from %s",
            split,
            len(self),
            self.base_path,
        )

    @staticmethod
    def _resolve_base_path(path: Path) -> Path:
        if (path / "ER").is_dir() or (path / "NR").is_dir():
            return path
        if (path / "train" / "ER").is_dir() or (path / "train" / "NR").is_dir():
            return path / "train"
        return path

    def _discover_files(self):
        files = []
        allowed_energies = (
            {int(energy) for energy in self.config.energies}
            if self.config.energies is not None
            else None
        )

        for recoil_type in self.config.recoil_types:
            recoil_path = self.base_path / recoil_type
            if not recoil_path.exists():
                logger.warning("H5 recoil path does not exist: %s", recoil_path)
                continue

            by_energy = {}
            for h5_path in sorted(recoil_path.glob(f"{recoil_type}_traces_energy_*_batch_*.h5")):
                match = self._FILENAME_RE.match(h5_path.name)
                if match is None:
                    continue
                energy = int(match.group("energy"))
                if allowed_energies is not None and energy not in allowed_energies:
                    continue
                by_energy.setdefault(energy, []).append(h5_path)

            for energy, energy_files in sorted(by_energy.items()):
                if self.config.max_h5_files_per_energy_recoil is not None:
                    energy_files = energy_files[
                        : self.config.max_h5_files_per_energy_recoil
                    ]
                for h5_path in energy_files:
                    files.append(
                        {
                            "energy": energy,
                            "recoil_type": recoil_type,
                            "path": h5_path,
                        }
                    )

        if not files:
            raise ValueError(f"No H5 batch files found under {self.base_path}")
        return files

    def _create_sample_index(self):
        for file_info in self._discover_files():
            try:
                with h5py.File(file_info["path"], "r") as handle:
                    if "events" not in handle or "traces" not in handle:
                        raise ValueError("missing events or traces dataset")
                    n_events = len(handle["events"])
                    traces = handle["traces"]
                    if traces.dtype != np.float16:
                        raise ValueError(f"traces dtype is {traces.dtype}, expected float16")
                    if traces.ndim != 3 or traces.shape[0] != n_events:
                        raise ValueError(
                            f"traces shape {traces.shape} does not match {n_events} events"
                        )
                    if (
                        self.config.expected_h5_events_per_file is not None
                        and n_events != self.config.expected_h5_events_per_file
                    ):
                        raise ValueError(
                            f"contains {n_events} events, expected "
                            f"{self.config.expected_h5_events_per_file}"
                        )
            except Exception as exc:
                raise ValueError(
                    f"Invalid H5 batch file {file_info['path']}: {exc}"
                ) from exc

            for event_idx in range(n_events):
                self.sample_index.append(
                    {
                        "energy": file_info["energy"],
                        "recoil_type": file_info["recoil_type"],
                        "path": file_info["path"],
                        "event_idx": event_idx,
                    }
                )

        if not self.sample_index:
            raise ValueError(f"No H5 samples found under {self.base_path}")

    def _create_train_val_splits(self):
        grouped = {}
        for sample_info in self.sample_index:
            key = (sample_info["energy"], sample_info["recoil_type"])
            grouped.setdefault(key, []).append(sample_info)

        split_samples = []
        np.random.seed(0)
        for (energy, recoil_type), samples in sorted(grouped.items()):
            indices = np.random.permutation(len(samples))
            train_end = int(self.config.train_split * len(indices))
            selected_indices = (
                indices[:train_end] if self.split == "train" else indices[train_end:]
            )
            split_samples.extend(samples[idx] for idx in selected_indices)
            print(
                f"Energy {energy}, Recoil {recoil_type}: "
                f"{len(selected_indices)} samples for {self.split} "
                f"(out of {len(samples)} total)"
            )

        if not split_samples:
            raise ValueError(f"No H5 samples found for split: {self.split}")
        return split_samples

    def verify_split_distribution(self):
        energy_distribution = {}
        recoil_distribution = {}
        for sample in self.split_samples:
            energy_distribution[sample["energy"]] = (
                energy_distribution.get(sample["energy"], 0) + 1
            )
            recoil_distribution[sample["recoil_type"]] = (
                recoil_distribution.get(sample["recoil_type"], 0) + 1
            )

        print(f"\n{self.split.upper()} H5 SPLIT VERIFICATION:")
        print(f"Energy levels: {sorted(energy_distribution.keys())}")
        print(f"Energy distribution: {energy_distribution}")
        print(f"Recoil distribution: {recoil_distribution}")
        return energy_distribution, recoil_distribution

    def __len__(self) -> int:
        return len(self.split_samples)

    def _get_handle(self, path: Path):
        path = str(path)
        handle = self._file_handles.pop(path, None)
        if handle is None:
            handle = h5py.File(path, "r")
        self._file_handles[path] = handle

        while len(self._file_handles) > self.config.max_open_h5_files:
            _, oldest_handle = self._file_handles.popitem(last=False)
            oldest_handle.close()
        return handle

    @staticmethod
    def _decode_recoil(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "decode"):
            return value.decode("utf-8")
        return str(value)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        sample_info = self.split_samples[idx]
        handle = self._get_handle(sample_info["path"])

        traces = handle["traces"][sample_info["event_idx"]]
        metadata_row = handle["events"][sample_info["event_idx"]]

        input_tensor = torch.from_numpy(traces).float()
        spatial_target = torch.tensor(
            [
                metadata_row["x"],
                metadata_row["y"],
                metadata_row["z"],
            ],
            dtype=torch.float32,
        )
        energy_target = torch.tensor(metadata_row["energy"], dtype=torch.float32)
        recoil_type = self._decode_recoil(metadata_row["type_recoil"])

        if input_tensor.size(-1) > self.config.max_seq_len:
            input_tensor = input_tensor[..., : self.config.max_seq_len]
        elif input_tensor.size(-1) < self.config.max_seq_len:
            pad_size = self.config.max_seq_len - input_tensor.size(-1)
            input_tensor = F.pad(input_tensor, (0, pad_size))

        if self.transform:
            input_tensor = self.transform(input_tensor)

        return input_tensor, spatial_target, energy_target, recoil_type

    def __del__(self):
        for handle in self._file_handles.values():
            try:
                handle.close()
            except Exception:
                pass


def collate_simple(batch):
    """Simple collate function for consistent channel counts"""
    inputs, spatial_targets, energy_targets, recoil_types = zip(*batch)
    
    return (
        torch.stack(inputs),
        torch.stack(spatial_targets),
        torch.stack(energy_targets),
        list(recoil_types)
    )

def create_dataloaders(
    data_config: DataConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
) -> Dict[str, DataLoader]:
    """Create dataloaders with proper train/val/test structure"""

    pin_memory = torch.cuda.is_available()

    dataset_cls = (
        H5BatchParticleReconstructionDataset
        if data_config.data_format == "h5_batch"
        else ParticleReconstructionDataset
    )
    datasets = {split: dataset_cls(data_config, split=split) for split in ["train", "val"]}

    dataloaders = {}
    for split, dataset in datasets.items():
        # Full-dataset validation evaluates only a bounded number of batches.
        # Shuffle validation too so those batches represent all energy/recoil
        # groups rather than always starting from the first sorted group.
        if split == "train" and distributed:
            sampler = DistributedSampler(dataset, shuffle=True, seed=0)
            shuffle = False
        else:
            sampler = None
            shuffle = split in {"train", "val"}

        generator = torch.Generator()
        generator.manual_seed(0 if split == "train" else 1)

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            generator=generator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=collate_simple,
        )

    return dataloaders
