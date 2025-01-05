from enum import Enum

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

class Band(Enum):
    DELTA = "DELTA"
    THETA = "THETA"
    ALPHA = "ALPHA"
    BETA = "BETA"
    GAMMA = "GAMMA"
    ALL = "ALL"


def get_band_range(band: Band) -> tuple[float, float]:
    """
    Get frequency range for a specific brain wave band.

    Args:
        band: Brain wave band.

    Returns:
        Tuple of (low_freq, high_freq) for the specified band.
    """
    if band == Band.DELTA:
        return 0.5, 4.0
    if band == Band.THETA:
        return 4.0, 8.0
    if band == Band.ALPHA:
        return 8.0, 13.0
    if band == Band.BETA:
        return 13.0, 30.0
    if band == Band.GAMMA:
        return 30.0, 100.0
    if band == Band.ALL:
        return 0.5, 100.0
    raise ValueError(f"Invalid band: {band}")


def compute_power(channel_data: np.ndarray, sampling_rate: int, band: Band) -> float:
    """
    Extract band power from EEG data.

    Args:
        channel_data: 1D numpy array of samples.
        sampling_rate: Sampling rate in Hz.
        band: Brain wave band.

    Returns:
        Band power as a float.
    """
    low_freq, high_freq = get_band_range(band)
    filtered = channel_data.copy()
    DataFilter.detrend(filtered, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(
        filtered,
        sampling_rate,
        low_freq,
        high_freq,
        4,
        FilterTypes.BUTTERWORTH.value,
        0
    )
    return np.sqrt(np.mean(np.square(filtered)))
