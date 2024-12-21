from neuros.board.board_interface import BoardInterface, ChannelConfig
from neuros.processing.simple_processing import extract_alpha_power
from neuros.output.audio_output import AudioOutput, AudioChannelConfig
from brainflow.board_shim import BoardIds
import time


def run_neurofeedback(duration_seconds: float = 60.0, window_size_ms: float = 50.0):
    """Run the neurofeedback system with audio output."""

    # Set up board
    channels = [ChannelConfig(i) for i in range(3)]  # Just use first 3 channels for now
    board = BoardInterface(BoardIds.SYNTHETIC_BOARD, channels)

    # Set up audio with harmonic frequencies
    base_freq = 256.0  # Middle C
    audio_configs = [
        AudioChannelConfig(0, base_freq),  # 256 Hz
        AudioChannelConfig(1, base_freq * 1.5),  # 384 Hz
        AudioChannelConfig(2, base_freq * 2.0)  # 512 Hz
    ]
    audio = AudioOutput(audio_configs)

    print(f"\nStarting neurofeedback session for {duration_seconds} seconds...")
    print("Press Ctrl+C to stop")

    try:
        with board, audio:  # Uses context managers for clean setup/teardown
            start_time = time.time()
            while (time.time() - start_time) < duration_seconds:
                # Get latest data window
                channel_data = board.get_data_window(int(window_size_ms))

                # Process each channel and update audio
                channel_values = {
                    idx: extract_alpha_power(data, board.sampling_rate)
                    for idx, data in channel_data.items()
                }

                audio.update(channel_values)

                # Small sleep to prevent CPU overload
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nSession stopped by user")

    print("Session complete")


def run_stats_collection():
    """Run the statistics collection (previous version)"""
    # ... (keep the existing collect_channel_stats function and its call)


if __name__ == "__main__":
    # Choose which mode to run
    run_neurofeedback(duration_seconds=60.0, window_size_ms=50.0)
    # or
    # run_stats_collection()