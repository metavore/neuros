import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations


def extract_alpha_power(data: np.ndarray, sampling_rate: int) -> float:
    """
    Extract alpha band (8-13 Hz) power from EEG data.

    Args:
        data: Raw EEG data array
        sampling_rate: Sampling rate in Hz

    Returns:
        Normalized power value between 0 and 1
    """
    # Make a copy to avoid modifying original data
    filtered = data.copy()

    # Remove DC offset
    DataFilter.detrend(filtered, DetrendOperations.CONSTANT.value)

    # Apply bandpass filter for alpha band
    DataFilter.perform_bandpass(
        filtered,
        sampling_rate,
        8.0,  # Lower cutoff
        13.0,  # Upper cutoff
        4,  # Filter order
        FilterTypes.BUTTERWORTH.value,
        0  # Filter mode
    )

    # Compute RMS power and normalize
    power = np.sqrt(np.mean(np.square(filtered)))

    # Simple normalization - you might want to adjust these bounds
    normalized = np.clip(power / 50.0, 0, 1)

    return float(normalized)


def compute_band_ratio(data: np.ndarray,
                       sampling_rate: int,
                       band1: tuple[float, float],
                       band2: tuple[float, float]) -> float:
    """
    Compute the ratio of power between two frequency bands.

    Args:
        data: Raw EEG data array
        sampling_rate: Sampling rate in Hz
        band1: Tuple of (low, high) frequencies for first band
        band2: Tuple of (low, high) frequencies for second band

    Returns:
        Ratio of band1 power to band2 power
    """

    def get_band_power(band: tuple[float, float]) -> float:
        filtered = data.copy()
        DataFilter.detrend(filtered, DetrendOperations.CONSTANT.value)

        DataFilter.perform_bandpass(
            filtered,
            sampling_rate,
            band[0],
            band[1],
            4,
            FilterTypes.BUTTERWORTH.value,
            0
        )

        return np.sqrt(np.mean(np.square(filtered)))

    power1 = get_band_power(band1)
    power2 = get_band_power(band2)

    # Avoid division by zero
    if power2 == 0:
        return 0.0

    ratio = power1 / power2
    return float(np.clip(ratio, 0, 1))
