from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_SENSOR_POSITIONS_PATH = _PROJECT_ROOT / "position_MMC_V2.dat"
_CACHE_DIR = _PROJECT_ROOT / "cache"

def compute_and_save_relative_positions(x: np.ndarray, output_dir: str | Path):
    relative_positions = x[None, :] - x[:, None]
    save_path = output_dir / "pos_diff.npy"
    np.save(save_path, relative_positions)

def main():
    sensor_positions = np.loadtxt(_SENSOR_POSITIONS_PATH, usecols=(1, 2, 3))
    compute_and_save_relative_positions(sensor_positions, _CACHE_DIR)
    logger.info("Sensor position differences successfully saved.")


if __name__ == "__main__":
    main()
