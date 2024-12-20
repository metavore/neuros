import pytest
import numpy as np
from unittest.mock import patch, Mock, call
from threading import Thread
import sounddevice as sd
from neuros.output.tone_generator import ToneGenerator, ToneConfig


class TestToneGenerator:
    def test_init_default_config(self):
        """Test initialization with default configuration"""
        with patch('sounddevice.query_devices') as mock_query:
            mock_query.return_value = [{'max_output_channels': 2, 'index': 0}]
            generator = ToneGenerator()
            assert generator.config.frequency == 440.0
            assert generator.config.sample_rate == 44100
            assert generator._phase == 0

    def test_init_custom_config(self):
        """Test initialization with custom configuration"""
        with patch('sounddevice.query_devices') as mock_query:
            mock_query.return_value = [{'max_output_channels': 2, 'index': 0}]
            config = ToneConfig(frequency=880.0, sample_rate=48000)
            generator = ToneGenerator(config)
            assert generator.config.frequency == 880.0
            assert generator.config.sample_rate == 48000

    def test_waveform_generation(self):
        """Test waveform generation for different types"""
        with patch('sounddevice.query_devices') as mock_query:
            mock_query.return_value = [{'max_output_channels': 2, 'index': 0}]
            generator = ToneGenerator()
            t = np.linspace(0, 2 * np.pi, 1000)

            # Test sine wave
            generator.config.waveform = 'sine'
            sine = generator._generate_waveform(t)
            assert np.allclose(sine[0], 0)  # Starts at 0
            assert np.allclose(sine[250], 1, atol=0.1)  # Peak around π/2

            # Test square wave
            generator.config.waveform = 'square'
            square = generator._generate_waveform(t)
            assert np.all(np.logical_or(np.isclose(square, 1), np.isclose(square, -1)))

            # Test sawtooth wave
            generator.config.waveform = 'sawtooth'
            saw = generator._generate_waveform(t)
            assert np.allclose(saw[0], 0, atol=0.1)
            assert np.any(saw > 0.9)  # Should reach near 1
            assert np.any(saw < -0.9)  # Should reach near -1

    def test_amplitude_bounds(self):
        """Test amplitude setting and bounds"""
        with patch('sounddevice.query_devices') as mock_query:
            mock_query.return_value = [{'max_output_channels': 2, 'index': 0}]
            config = ToneConfig(min_amplitude=0.1, max_amplitude=0.8)
            generator = ToneGenerator(config)

            # Test minimum bound
            generator.set_amplitude(-1)
            assert generator.amplitude == 0.1

            # Test maximum bound
            generator.set_amplitude(2)
            assert generator.amplitude == 0.8

            # Test normal value
            generator.set_amplitude(0.5)
            expected = 0.1 + (0.8 - 0.1) * 0.5
            assert np.isclose(generator.amplitude, expected)

    def test_device_selection(self):
        """Test device selection and fallback behavior"""
        with patch('sounddevice.query_devices') as mock_query:
            mock_query.return_value = [{
                'name': 'Test Device',
                'max_output_channels': 2,
                'index': 0,
                'default_samplerate': 44100
            }]

            # Test default device selection
            generator = ToneGenerator()
            assert generator.config.device == 0

            # Test fallback when invalid device specified
            config = ToneConfig(device=999)  # Non-existent device
            generator = ToneGenerator(config)
            assert generator.config.device == 0  # Should fall back to default

    def test_stream_lifecycle(self):
        """Test stream start/stop behavior"""
        with patch('sounddevice.query_devices') as mock_query, \
                patch('sounddevice.OutputStream') as mock_stream:
            mock_query.return_value = [{'max_output_channels': 2, 'index': 0}]
            generator = ToneGenerator()

            # Test start
            generator.start()
            mock_stream.assert_called_once()
            mock_stream.return_value.start.assert_called_once()

            # Test stop
            generator.stop()
            mock_stream.return_value.stop.assert_called_once()
            mock_stream.return_value.close.assert_called_once()

    def test_thread_safety(self):
        """Test thread-safe amplitude changes"""
        with patch('sounddevice.query_devices') as mock_query:
            mock_query.return_value = [{'max_output_channels': 2, 'index': 0}]
            generator = ToneGenerator()
            num_threads = 10
            iterations = 100

            def change_amplitude():
                for _ in range(iterations):
                    generator.set_amplitude(np.random.random())

            threads = [Thread(target=change_amplitude) for _ in range(num_threads)]

            # Start all threads
            for thread in threads:
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Check amplitude is within bounds
            assert 0 <= generator.amplitude <= 1

    def test_callback_phase_continuity(self):
        """Test phase continuity between callbacks"""
        with patch('sounddevice.query_devices') as mock_query:
            mock_query.return_value = [{'max_output_channels': 2, 'index': 0}]
            generator = ToneGenerator(ToneConfig(frequency=440))
            frames = 100

            # First callback
            outdata1 = np.zeros((frames, 1))
            generator._audio_callback(outdata1, frames, {}, None)
            phase1 = generator._phase

            # Second callback
            outdata2 = np.zeros((frames, 1))
            generator._audio_callback(outdata2, frames, {}, None)
            phase2 = generator._phase

            # Phase should advance by frames/sample_rate
            expected_phase_diff = frames / generator.config.sample_rate
            assert np.isclose(phase2 - phase1, expected_phase_diff)

    def test_error_handling(self):
        """Test error handling in audio callback"""
        with patch('sounddevice.query_devices') as mock_query, \
                patch('sounddevice.OutputStream') as mock_stream, \
                patch('builtins.print') as mock_print:
            mock_query.return_value = [{'max_output_channels': 2, 'index': 0}]
            generator = ToneGenerator()

            # Simulate error status in callback
            error_status = Mock()
            error_status.output_underflow = True

            outdata = np.zeros((100, 1))
            generator._audio_callback(outdata, 100, {}, error_status)
            mock_print.assert_called_with("Status: CallbackFlags.output_underflow")

