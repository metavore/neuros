from enum import Enum
import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations


class Band(Enum):
    DELTA = (0.5, 4.0)
    THETA = (4.0, 8.0)
    ALPHA = (8.0, 13.0)
    BETA = (13.0, 30.0)
    GAMMA = (30.0, 100.0)
    ALL = (0.5, 100.0)


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
    low_freq, high_freq = band.value
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
