import logging
from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np
from contextlib import contextmanager
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for data windowing"""
    window_ms: float  # Window size in milliseconds
    overlap_ms: float = 0.0  # Overlap between windows

    def to_samples(self, sample_rate: int) -> tuple[int, int]:
        """Convert time-based config to sample counts"""
        window_samples = int(sample_rate * self.window_ms / 1000)
        overlap_samples = int(sample_rate * self.overlap_ms / 1000)
        if overlap_samples >= window_samples:
            raise ValueError("Overlap must be less than window size")
        return window_samples, overlap_samples


@contextmanager
def board_stream(board_id: int = BoardIds.SYNTHETIC_BOARD,
                 params: Optional[BrainFlowInputParams] = None) -> BoardShim:
    """
    Create and manage a board connection.

    Args:
        board_id: BrainFlow board identifier
        params: Optional board parameters

    Yields:
        Connected BoardShim instance

    Raises:
        BrainFlowError: If board initialization or streaming fails
    """
    board = BoardShim(board_id, params or BrainFlowInputParams())
    try:
        board.prepare_session()
        board.start_stream()
        yield board
    except Exception as e:
        logger.error(f"Error during board operation: {e}")
        raise
    finally:
        try:
            if board.is_prepared():
                board.stop_stream()
                board.release_session()
        except Exception as e:
            logger.error(f"Error during board cleanup: {e}")


def stream_windows(board: BoardShim, config: WindowConfig) -> Iterator[np.ndarray]:
    sample_rate = board.get_sampling_rate(board_id=board.board_id)
    window_samples, overlap_samples = config.to_samples(sample_rate)
    new_samples = window_samples - overlap_samples

    # Get the EEG channel indices
    eeg_channels = board.get_eeg_channels(board_id=board.board_id)

    # Create buffer just for EEG channels
    ring_buffer = np.zeros((len(eeg_channels), window_samples), dtype=np.float32)
    filled_samples = 0

    try:
        while True:
            try:
                new_data = board.get_current_board_data(new_samples)
                if new_data.size == 0 or new_data.shape[1] != new_samples:
                    continue  # Skip if we don't have enough samples yet, e.g. at start

                # Extract just the EEG channels from the new data
                eeg_data = new_data[eeg_channels]

                ring_buffer[:, :-new_samples] = ring_buffer[:, new_samples:]
                ring_buffer[:, -new_samples:] = eeg_data

                filled_samples += new_data.shape[1]

                if filled_samples >= window_samples:
                    yield ring_buffer.copy()

            except Exception as e:
                logger.error(f"Error getting board data: {e}")
                continue

    except GeneratorExit:
        logger.info("Stopping window streaming")


def main() -> None:
    """Example usage"""
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)

    try:
        with board_stream() as board:
            logger.info(f"Board ready - sample rate: {board.get_sampling_rate(board_id=BoardIds.SYNTHETIC_BOARD)} Hz")
            logger.info("Press Ctrl+C to stop...")

            for window in stream_windows(board, config):
                print(f"Got window shape: {window.shape}")

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
