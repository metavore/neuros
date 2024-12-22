from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Callable, Any
import numpy as np
import threading
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Global lock for thread-safe board operations
_board_lock = threading.Lock()


@contextmanager
def board_stream(board_id: int = BoardIds.SYNTHETIC_BOARD) -> Iterator[BoardShim]:
    """
    Context manager for safe board streaming operations.

    Usage:
        with board_stream() as board:
            data = board.get_current_board_data(100)
    """
    board = BoardShim(board_id, BrainFlowInputParams())
    try:
        board.prepare_session()
        board.start_stream()
        yield board
    finally:
        with _board_lock:
            if board.is_prepared():
                board.stop_stream()
                board.release_session()


@dataclass
class ProcessorConfig:
    """Configuration for the stream processor"""
    window_size_ms: float = 50.0  # Size of processing window in milliseconds
    overlap_ms: float = 0.0  # Overlap between windows in milliseconds


class StreamProcessor:
    """
    Processes streaming data in a separate thread.

    Usage:
        def process_data(data: np.ndarray):
            # data shape is (channels, samples)
            print(f"Processing {data.shape[1]} samples")

        with board_stream() as board:
            processor = StreamProcessor(board)
            processor.start(process_data)
            time.sleep(5)  # Process for 5 seconds
    """

    def __init__(self, board: BoardShim, config: Optional[ProcessorConfig] = None):
        self.board = board
        self.config = config or ProcessorConfig()

        # Convert ms to samples
        sample_rate = BoardShim.get_sampling_rate(board.get_board_id())
        self.window_samples = int(sample_rate * self.config.window_size_ms / 1000)
        self.overlap_samples = int(sample_rate * self.config.overlap_ms / 1000)

        # Processing thread management
        self._process_thread: Optional[threading.Thread] = None
        self._running = False
        self._on_data: Optional[Callable[[np.ndarray], Any]] = None

    def start(self, on_data: Callable[[np.ndarray], Any]) -> None:
        """Start processing data from the stream."""
        if self._process_thread is not None:
            return

        self._on_data = on_data
        self._running = True
        self._process_thread = threading.Thread(target=self._process_loop)
        self._process_thread.daemon = True
        self._process_thread.start()

    def stop(self) -> None:
        """Stop processing and clean up."""
        self._running = False
        if self._process_thread is not None:
            self._process_thread.join()
            self._process_thread = None

    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Get latest data window - thread-safe
                with _board_lock:
                    data = self.board.get_current_board_data(self.window_samples)

                # Only process if we have exactly a full window
                if data.shape[1] == self.window_samples and self._on_data is not None:
                    try:
                        self._on_data(data)
                    except Exception as e:
                        print(f"Error in data callback: {e}")

                # Sleep for window duration minus overlap
                sleep_time = max(0.001,
                                 (self.window_samples - self.overlap_samples)
                                 / BoardShim.get_sampling_rate(self.board.get_board_id()))
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Error in process loop: {e}")
                time.sleep(0.1)


def example_usage():
    """Example showing how to use the streaming interface."""

    def process_data(data: np.ndarray):
        channels, samples = data.shape
        print(f"Processing {channels} channels, {samples} samples each")

    # Use both context managers together
    with board_stream(board_id=BoardIds.SYNTHETIC_BOARD) as board:
        # Create processor with 100ms windows, 50% overlap
        config = ProcessorConfig(window_size_ms=100.0, overlap_ms=50.0)
        processor = StreamProcessor(board, config)

        try:
            # Start processing
            processor.start(process_data)
            # Run for 5 seconds
            time.sleep(5)
        finally:
            processor.stop()


if __name__ == "__main__":
    example_usage()