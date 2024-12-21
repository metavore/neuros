from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import time
from neuros.output.tone_generator import ToneGenerator, ToneConfig


@dataclass
class AudioChannelConfig:
    """Configuration for a single audio output channel"""
    channel_index: int
    frequency: float
    min_amplitude: float = 0.0
    max_amplitude: float = 1.0
    waveform: str = 'sine'


class AudioOutput:
    """
    Manages multiple tones and their modulation based on EEG channel data.
    Uses existing ToneGenerator for actual sound production.
    """

    def __init__(self, channel_configs: List[AudioChannelConfig]):
        self.channel_configs = channel_configs
        self.tone_generators: Dict[int, ToneGenerator] = {}
        self._setup_generators()

    def _setup_generators(self):
        """Initialize tone generators for each channel"""
        for config in self.channel_configs:
            tone_config = ToneConfig(
                frequency=config.frequency,
                min_amplitude=config.min_amplitude,
                max_amplitude=config.max_amplitude,
                waveform=config.waveform
            )
            self.tone_generators[config.channel_index] = ToneGenerator(tone_config)

    def start(self):
        """Start all tone generators with startup sequence"""
        print("Starting audio output...")
        self._play_startup_sequence()

    def stop(self):
        """Stop all tone generators"""
        for generator in self.tone_generators.values():
            generator.stop()

    def update(self, channel_values: Dict[int, float]):
        """
        Update amplitudes based on new channel values.

        Args:
            channel_values: Dictionary mapping channel indices to their values (0-1 range)
        """
        for channel_idx, value in channel_values.items():
            if channel_idx in self.tone_generators:
                self.tone_generators[channel_idx].set_amplitude(value)

    def _play_startup_sequence(self):
        """Play a brief startup sequence to verify audio output"""
        print("Playing startup sequence...")

        # Start all generators at zero amplitude
        for generator in self.tone_generators.values():
            generator.start()
            generator.set_amplitude(0.0)

        # Play each tone briefly in sequence
        for config in self.channel_configs:
            print(f"Testing tone {config.frequency:.1f} Hz...")
            generator = self.tone_generators[config.channel_index]

            # Fade in
            for amp in np.linspace(0, 0.5, 20):
                generator.set_amplitude(amp)
                time.sleep(0.01)

            time.sleep(0.3)  # Hold

            # Fade out
            for amp in np.linspace(0.5, 0, 20):
                generator.set_amplitude(amp)
                time.sleep(0.01)

            time.sleep(0.1)  # Brief pause between tones

        print("Startup sequence complete")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False