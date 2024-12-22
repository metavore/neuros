import logging

import pytest
import numpy as np
import time
from brainflow.board_shim import BoardIds
from neuros.board.stream_processor import board_stream, StreamProcessor, ProcessorConfig

logging.basicConfig(level=logging.DEBUG)

def test_board_stream_context():
    """Test basic context manager functionality."""
    with board_stream() as board:
        assert board.is_prepared()
        # Try getting some data to verify streaming is working
        data = board.get_current_board_data(100)
        assert data.size > 0

    # Board should be cleaned up after context
    assert not board.is_prepared()


def test_board_stream_exception_cleanup():
    """Test cleanup happens even when exception occurs."""
    with pytest.raises(ValueError):
        with board_stream() as board:
            assert board.is_prepared()
            raise ValueError("Test exception")

    # Board should still be cleaned up
    assert not board.is_prepared()


def test_stream_processor_basic():
    """Test StreamProcessor delivers data to callback."""
    received_data = []

    def callback(data: np.ndarray):
        received_data.append(data)

    with board_stream() as board:
        processor = StreamProcessor(board, ProcessorConfig(window_size_ms=100.0))
        processor.start(callback)

        # Wait for some data
        time.sleep(0.5)
        processor.stop()

    # Should have received some data
    assert len(received_data) > 0
    # Data should be in correct shape (channels x samples)
    assert received_data[0].ndim == 2
    assert received_data[0].shape[1] > 0  # At least some samples


def test_stream_processor_window_size():
    """Test processor respects window size configuration."""
    received_sizes = []

    def callback(data: np.ndarray):
        received_sizes.append(data.shape[1])

    # Use 200ms windows
    window_ms = 200.0
    with board_stream() as board:
        sample_rate = board.get_sampling_rate(board_id=BoardIds.SYNTHETIC_BOARD)
        expected_samples = int(sample_rate * window_ms / 1000)

        processor = StreamProcessor(board, ProcessorConfig(window_size_ms=window_ms))
        processor.start(callback)

        # Wait for a few windows
        time.sleep(1.0)
        processor.stop()

    # Should have received some data
    assert len(received_sizes) > 0
    logging.debug(f"Received window sizes: {received_sizes}")

    # Check window sizes (allow small variation due to timing)
    for size in received_sizes:
        logging.debug(f"Window size: {size}")
        assert abs(size - expected_samples) <= 2


def test_stream_processor_overlap():
    """Test processor handles window overlap correctly."""
    timestamps = []

    def callback(data: np.ndarray):
        timestamps.append(time.time())

    window_ms = 100.0
    overlap_ms = 50.0

    with board_stream() as board:
        processor = StreamProcessor(
            board,
            ProcessorConfig(window_size_ms=window_ms, overlap_ms=overlap_ms)
        )
        processor.start(callback)

        # Wait for several windows
        time.sleep(0.5)
        processor.stop()

    # Check timing between callbacks
    intervals = np.diff(timestamps)
    expected_interval = (window_ms - overlap_ms) / 1000

    # Allow 20% timing variation (synthetic board can be wonky)
    assert all(abs(i - expected_interval) < expected_interval * 0.2
               for i in intervals)


def test_stream_processor_multiple_starts():
    """Test processor handles multiple start/stop cycles."""
    received_counts = []

    def callback(data: np.ndarray):
        received_counts.append(data.shape[1])

    with board_stream() as board:
        processor = StreamProcessor(board)

        # Start and stop multiple times
        for _ in range(3):
            processor.start(callback)
            time.sleep(0.2)
            processor.stop()
            time.sleep(0.1)

    # Should have received data in each cycle
    assert len(received_counts) > 0


def test_stream_processor_error_handling():
    """Test processor handles callback errors gracefully."""
    success_count = 0
    error_thrown = False

    def callback(data: np.ndarray):
        nonlocal success_count, error_thrown
        success_count += 1  # Increment first

        # Only throw the error once
        if success_count == 2 and not error_thrown:
            error_thrown = True
            raise ValueError("Test error")

    with board_stream() as board:
        processor = StreamProcessor(board)
        processor.start(callback)

        # Wait for a few callbacks
        time.sleep(0.5)
        processor.stop()

    # Should have continued past the error
    assert success_count > 2  # We should get more callbacks after the error


if __name__ == "__main__":
    pytest.main(["-v", __file__])
