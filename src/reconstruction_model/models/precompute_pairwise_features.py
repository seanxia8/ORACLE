from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).parent
_SENSOR_POSITIONS_PATH = _PACKAGE_ROOT / "position_MMC_V2.dat"
_CACHE_DIR = _PACKAGE_ROOT / "cache"

def compute_and_save_edge_features(x: np.ndarray, output_dir: str | Path):
    # (1, C, 3) - (C, 1, 3) -> (C, C, 3) - (C, C, 3) -> (C, C, 3)
    relative_positions = x[None, :] - x[:, None]
    # [0,...0,1,...1]
    array_ids = np.array([0] * 19 + [1] * 37)
    # (C, N) == (C, 1) -> (C, C)
    same_array = (array_ids[None, :] == array_ids[:, None]).astype(float)
    edge_feats = np.concatenate(
        [
            relative_positions,
            same_array[:, :, None],
        ],
        axis=-1,
    )
    save_path = output_dir / "edge_feats.npy"
    np.save(save_path, edge_feats)

def main():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sensor_positions = np.loadtxt(_SENSOR_POSITIONS_PATH, usecols=(1, 2, 3))
    compute_and_save_edge_features(sensor_positions, _CACHE_DIR)
    logger.info("Sensor position differences successfully saved.")


if __name__ == "__main__":
    main()
