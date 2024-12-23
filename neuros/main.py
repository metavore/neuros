import logging
from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np
from contextlib import contextmanager
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        logger.error(f"Error during board operation: {str(e)}")
        raise
    finally:
        try:
            if board.is_prepared():
                board.stop_stream()
                board.release_session()
        except Exception as e:
            logger.error(f"Error during board cleanup: {str(e)}")
            # Don't re-raise during cleanup


def stream_windows(board: BoardShim, config: WindowConfig) -> Iterator[np.ndarray]:
    """
    Generate windows of EEG data from board stream.

    This generator yields windows of data continuously from the board stream.
    Each window contains data from all channels (both EEG and non-EEG).
    Windows can overlap based on configuration, useful for frequency analysis.

    Args:
        board: Connected and streaming BoardShim instance
        config: Window configuration for size and overlap

    Yields:
        Windows of data as numpy arrays with shape (channels, samples)
        where channels includes all board channels (both EEG and non-EEG)
        and samples is determined by the window_ms configuration

    Note:
        The first few windows may be delayed as the board begins
        collecting data. The generator will continue yielding
        indefinitely until iteration is stopped.
    """
    # Convert ms-based config to samples
    sample_rate = board.get_sampling_rate(board_id=board.board_id)
    window_samples, overlap_samples = config.to_samples(sample_rate)

    # Buffer for overlap samples
    buffer = np.array([])

    # Placeholder for new data
    new_data = np.array([])

    try:
        while True:
            # Calculate how many new samples we need
            if buffer.size == 0:
                samples_needed = window_samples
            else:
                samples_needed = window_samples - overlap_samples

            # Initialize new_data as empty but with correct shape
            new_data = np.array([])

            try:
                # Get new data
                new_data = board.get_current_board_data(samples_needed)
                if new_data.size == 0:
                    # No new data available yet
                    continue

                # Add to buffer
                buffer = (np.concatenate([buffer, new_data], axis=1) if buffer.size > 0 else new_data)

                # Once we have enough samples, yield a window
                if buffer.shape[1] >= window_samples:
                    # Extract the window
                    window = buffer[:, :window_samples]
                    yield window

                    # Update buffer for next iteration
                    if overlap_samples > 0:
                        buffer = buffer[:, -overlap_samples:]
                    else:
                        buffer = np.array([])

            except Exception as e:
                logger.error(f"Error getting board data: {str(e)}")
                continue  # Try again on next iteration

    except GeneratorExit:
        logger.info("Stopping window streaming")

        # Add to buffer
        buffer = (np.concatenate([buffer, new_data], axis=1) if buffer.size > 0 else new_data)

        # Once we have enough samples, yield a window
        if buffer.shape[1] >= window_samples:
            # Extract the window
            window = buffer[:, :window_samples]
            yield window

    except Exception as e:
        logger.error(f"Error during window streaming: {str(e)}")
        raise


def main() -> None:
    """Example usage"""
    # Configuration for 500ms windows with 250ms overlap
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)

    try:
        with board_stream() as board:
            logger.info(f"Board ready - sample rate: {board.get_sampling_rate(board_id=BoardIds.SYNTHETIC_BOARD)} Hz")
            logger.info("Press Ctrl+C to stop...")

            # Process windows until interrupted
            for i, window in enumerate(stream_windows(board, config)):
                channels: int
                samples: int
                channels, samples = window.shape
                print(f"Window {i}: {channels} channels, {samples} samples")

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()
