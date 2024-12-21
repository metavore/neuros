from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams


@dataclass
class ChannelConfig:
    """Configuration for a single EEG channel"""
    index: int
    label: Optional[str] = None
    enabled: bool = True


class BoardInterface:
    """
    Handles all direct board interaction and basic data acquisition.
    Wraps BrainFlow functionality in a clean, board-agnostic interface.
    """

    def __init__(self, board_id: int, channels: List[ChannelConfig]):
        self.board_id = board_id
        self.channels = channels
        self._board = None
        self._sampling_rate = None
        self._eeg_channels = None

    def start(self):
        """Initialize and start the board"""
        params = BrainFlowInputParams()
        self._board = BoardShim(self.board_id, params)
        self._sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self._eeg_channels = BoardShim.get_eeg_channels(self.board_id)

        self._board.prepare_session()
        self._board.start_stream()

    def stop(self):
        """Stop and release the board"""
        if self._board and self._board.is_prepared():
            self._board.stop_stream()
            self._board.release_session()

    def get_data_window(self, window_size_ms: int) -> Dict[int, np.ndarray]:
        """
        Get the latest window of data for each enabled channel.

        Args:
            window_size_ms: Size of the window in milliseconds

        Returns:
            Dictionary mapping channel indices to their data arrays
        """
        if not self._board:
            raise RuntimeError("Board not started")

        samples = int(self._sampling_rate * window_size_ms / 1000)
        data = self._board.get_current_board_data(samples)

        # Create a dictionary of enabled channel data
        channel_data = {}
        for channel in self.channels:
            if channel.enabled:
                idx = self._eeg_channels[channel.index]
                channel_data[channel.index] = data[idx]

        return channel_data

    def set_board_id(self, board_id: int):
        """Set the board ID - use BoardIds enum from BrainFlow"""
        self.board_id = board_id

    @property
    def sampling_rate(self) -> int:
        """Get the board's sampling rate"""
        if self._sampling_rate is None:
            raise RuntimeError("Board not initialized")
        return self._sampling_rate

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False