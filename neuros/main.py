import logging
import time
from brainflow.board_shim import BoardIds
from neuros.window_stream import WindowConfig, create_board_stream, stream_windows
from neuros.process_data import Band, compute_power, get_band_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Example usage demonstrating window streaming with synthetic board"""
    config = WindowConfig(window_ms=550.0, overlap_ms=225.0)

    try:
        with create_board_stream(board_id=BoardIds.SYNTHETIC_BOARD) as board:
            sample_rate = board.get_sampling_rate(board_id=board.board_id)
            logger.info(f"Board ready - sample rate: {sample_rate} Hz")
            logger.info("Press Ctrl+C to stop...")

            for window in stream_windows(board, config):
                # Get data just for first channel
                channel_1 = window[0]
                logger.debug(f"Window shape: {window.shape}")
                alpha_power = compute_power(channel_1, sample_rate, Band.ALPHA)
                total_power = compute_power(channel_1, sample_rate, Band.ALL)
                alpha_ratio = alpha_power / (total_power + 1e-10)
                logger.info(f"Alpha power: {alpha_power:.2f}, Alpha ratio: {alpha_ratio:.2f}")

                # Small delay to prevent CPU overload
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
