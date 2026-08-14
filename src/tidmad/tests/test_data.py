from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np

from tidmad.data import read_channel_pair


def test_read_channel_pair_uses_full_contract_paths(tmp_path):
    path = tmp_path / "abra_training_0000.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset(
            "timeseries/channel0001/timeseries",
            data=np.asarray([-128, -127, -126, -125], dtype=np.int8),
        )
        fh.create_dataset(
            "timeseries/channel0002/timeseries",
            data=np.asarray([-120, -119, -118, -117], dtype=np.int8),
        )
    contract = SimpleNamespace(
        squid_dataset="timeseries/channel0001/timeseries",
        reference_dataset="timeseries/channel0002/timeseries",
    )
    squid, reference = read_channel_pair(path, contract, start=1, length=2)
    np.testing.assert_array_equal(squid, [1.0, 2.0])
    np.testing.assert_array_equal(reference, [9.0, 10.0])
