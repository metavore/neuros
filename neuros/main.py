from neuros.board.board_interface import BoardInterface, ChannelConfig
from neuros.processing.simple_processing import extract_alpha_power, compute_band_ratio
from brainflow.board_shim import BoardIds
import numpy as np
import time
from collections import defaultdict


def collect_channel_stats(duration_seconds: float = 60.0, window_size_ms: float = 50.0):
    """
    Collect statistics for all channels over the specified duration.
    """
    # Initialize board with all channels enabled
    channels = [ChannelConfig(i) for i in range(8)]  # 8 channels
    board = BoardInterface(BoardIds.SYNTHETIC_BOARD, channels)

    # Storage for our statistics
    alpha_powers = defaultdict(list)
    alpha_theta_ratios = defaultdict(list)

    print(f"Starting data collection for {duration_seconds} seconds...")

    try:
        with board:  # Uses context manager for clean start/stop
            start_time = time.time()
            while (time.time() - start_time) < duration_seconds:
                # Get data window for all channels
                channel_data = board.get_data_window(int(window_size_ms))

                # Process each channel
                for channel_idx, data in channel_data.items():
                    # Calculate alpha power
                    power = extract_alpha_power(data, board.sampling_rate)
                    alpha_powers[channel_idx].append(power)

                    # Calculate alpha/theta ratio
                    ratio = compute_band_ratio(
                        data,
                        board.sampling_rate,
                        (8, 13),  # Alpha band
                        (4, 8)  # Theta band
                    )
                    alpha_theta_ratios[channel_idx].append(ratio)

                # Small sleep to prevent CPU overload
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nData collection interrupted by user")

    # Print statistics for each channel
    print("\nChannel Statistics:")
    print("-" * 50)

    for channel_idx in sorted(alpha_powers.keys()):
        powers = np.array(alpha_powers[channel_idx])
        ratios = np.array(alpha_theta_ratios[channel_idx])

        print(f"\nChannel {channel_idx}:")
        print(f"  Alpha Power:")
        print(f"    Mean: {powers.mean():.3f}")
        print(f"    Std:  {powers.std():.3f}")
        print(f"    Min:  {powers.min():.3f}")
        print(f"    Max:  {powers.max():.3f}")
        print(f"  Alpha/Theta Ratio:")
        print(f"    Mean: {ratios.mean():.3f}")
        print(f"    Std:  {ratios.std():.3f}")
        print(f"    Min:  {ratios.min():.3f}")
        print(f"    Max:  {ratios.max():.3f}")


if __name__ == "__main__":
    collect_channel_stats(duration_seconds=60.0, window_size_ms=50.0)