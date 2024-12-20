import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import Optional, Dict, List
import threading
import time


@dataclass
class ToneConfig:
    """Configuration for tone generation"""
    frequency: float = 440.0
    min_amplitude: float = 0.0
    max_amplitude: float = 1.0
    sample_rate: int = 44100
    waveform: str = 'sine'
    device: Optional[int] = None


class ToneGenerator:
    """
    A configurable tone generator with real-time modulation capabilities.

    Features:
    - Multiple waveform types (sine, square, sawtooth, triangle)
    - Configurable output device
    - Thread-safe amplitude modulation
    """

    def __init__(self, config: Optional[ToneConfig] = None):
        """Initialize the tone generator with given or default configuration"""
        self.config = config or ToneConfig()
        self.amplitude = 0.0
        self.stream: Optional[sd.OutputStream] = None
        self._lock = threading.Lock()
        self._phase = 0
        self._setup_device()

    def _setup_device(self):
        """Validate and set up audio device"""
        try:
            # Try to get the default output device first
            default_device = sd.query_devices(kind='output')
            self.config.device = default_device['index']
        except:
            # Fall back to manual device selection if that fails
            devices = sd.query_devices()
            valid_devices = [dev for dev in devices if dev['max_output_channels'] > 0]

            if not valid_devices:
                raise RuntimeError("No valid output devices found")

            if self.config.device is not None:
                # Check if specified device exists and supports output
                if not any(dev['index'] == self.config.device for dev in valid_devices):
                    print(f"Warning: Specified device {self.config.device} not found")
                    self.config.device = valid_devices[0]['index']
            else:
                # Use first valid output device
                self.config.device = valid_devices[0]['index']

    @staticmethod
    def list_devices() -> List[Dict]:
        """List all available audio output devices"""
        devices = sd.query_devices()
        return [
            {
                'index': dev['index'],
                'name': dev['name'],
                'channels': dev['max_output_channels'],
                'default_samplerate': dev['default_samplerate']
            }
            for dev in devices
            if dev['max_output_channels'] > 0
        ]

    def _generate_waveform(self, t):
        """Generate waveform based on type"""
        if self.config.waveform == 'sine':
            return np.sin(t)
        elif self.config.waveform == 'square':
            # Ensure strict -1 or 1 values
            return np.where(np.sin(t) >= 0, 1.0, -1.0)
        elif self.config.waveform == 'sawtooth':
            return 2 * (t / (2 * np.pi) - np.floor(0.5 + t / (2 * np.pi)))
        elif self.config.waveform == 'triangle':
            return 2 * np.abs(2 * (t / (2 * np.pi) - np.floor(0.5 + t / (2 * np.pi)))) - 1
        else:
            raise ValueError(f"Unsupported waveform type: {self.config.waveform}")

    def _audio_callback(self, outdata: np.ndarray, frames: int,
                        time_info: Dict, status: Optional[sd.CallbackFlags]):
        """Generate audio data for the sound card"""
        if status:
            if status.output_underflow:
                print("Status: CallbackFlags.output_underflow")
            else:
                print(f"Status: {status}")

        with self._lock:
            current_amplitude = self.amplitude

        # Generate continuous waveform
        t = 2 * np.pi * self.config.frequency * \
            (np.arange(frames) / self.config.sample_rate + self._phase)

        # Update phase for next callback
        self._phase += frames / self.config.sample_rate

        # Generate and apply amplitude
        output = current_amplitude * self._generate_waveform(t)
        outdata[:, 0] = output.astype(np.float32)

    def start(self):
        """Start the audio stream"""
        if self.stream is not None:
            return

        self.stream = sd.OutputStream(
            channels=1,
            callback=self._audio_callback,
            samplerate=self.config.sample_rate,
            device=self.config.device
        )
        self.stream.start()

    def stop(self):
        """Stop the audio stream"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def set_amplitude(self, value: float):
        """Thread-safe amplitude adjustment"""
        with self._lock:
            normalized = np.clip(value, 0, 1)
            self.amplitude = (self.config.max_amplitude - self.config.min_amplitude) * \
                             normalized + self.config.min_amplitude


def demo():
    """Demonstrate various features of the ToneGenerator"""
    print("\nAvailable output devices:")
    for device in ToneGenerator.list_devices():
        print(f"Index: {device['index']}, Name: {device['name']}")

    # Create custom configuration
    config = ToneConfig(
        frequency=440,
        min_amplitude=0.1,
        max_amplitude=0.8,
        waveform='sine'
    )

    try:
        print("\nStarting tone generator...")
        print("Will demonstrate different waveforms")
        print("Press Ctrl+C to stop")

        for waveform in ['sine', 'square', 'triangle', 'sawtooth']:
            print(f"\nPlaying {waveform} wave")
            config.waveform = waveform
            tone = ToneGenerator(config)
            tone.start()

            # Modulate amplitude
            for i in range(20):
                time.sleep(0.1)
                amplitude = 0.45 + 0.35 * np.sin(time.time() * 2)
                tone.set_amplitude(amplitude)
                if i % 5 == 0:  # Print every 5th value
                    print(f"Amplitude: {amplitude:.2f}")

            tone.stop()
            time.sleep(0.5)  # Brief pause between waveforms

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Done!")


if __name__ == "__main__":
    demo()