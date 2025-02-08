import logging
import time
from brainflow.board_shim import BoardIds
from neuros.eeg_reader import WindowConfig, create_board_stream, stream_windows, Band, compute_power
from neuros.tone_generator import ToneGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Example usage demonstrating window streaming with synthetic board"""
    config = WindowConfig(window_ms=550.0, overlap_ms=225.0)
    max_alpha_ratio = 0.0  # Only one channel, one ratio for MVP
    windows_to_skip = 500  # Number of windows to skip, to allow for stabilization
    cpu_delay = 0.005  # Small delay between iterations to prevent CPU overload
    tone = ToneGenerator()

    try:
        with create_board_stream(board_id=BoardIds.SYNTHETIC_BOARD) as board:
            sample_rate = board.get_sampling_rate(board_id=board.board_id)
            logger.info(f"Board ready - sample rate: {sample_rate} Hz")
            logger.info("Press Ctrl+C to stop...")

            window_iterator = stream_windows(board, config)

            # Skip the first windows_to_skip windows
            for _ in range(windows_to_skip):
                next(window_iterator)

            for window in window_iterator:
                # Get data just for first channel
                channel_1 = window[0]
                alpha_power = compute_power(channel_1, sample_rate, Band.ALPHA)
                total_power = compute_power(channel_1, sample_rate, Band.ALL)
                alpha_ratio = alpha_power / (total_power + 1e-10)

                # Update the maximum alpha_ratio
                if alpha_ratio > max_alpha_ratio:
                    logger.info(f"New real max alpha ratio: {alpha_ratio:.2f}")
                    max_alpha_ratio = alpha_ratio

                normalized_alpha_ratio = alpha_ratio / max_alpha_ratio
                logger.info(f"Normalized alpha ratio: {normalized_alpha_ratio:.2f}")
                tone.set_velocity(normalized_alpha_ratio)

                # Small delay to prevent CPU overload
                time.sleep(cpu_delay)

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise
    finally:
        tone.cleanup()


if __name__ == "__main__":
    main()
