from fluidsynth import Synth


class ToneGenerator:
    """FluidSynth-based tone generator with continuous tones and adjustable volume."""

    def __init__(self):
        """Initialize FluidSynth with organ sound"""
        self._soundfont_path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
        self._instrument = 19  # Organ
        self._current_note = 60  # Middle C

        self._is_playing = False
        self._current_velocity = 64  # MIDI velocity aka amplitude

        # Initialize synthesizer
        self.synth = Synth()

        # Configure synthesizer
        self.synth.setting("synth.gain", 1.0)
        self.synth.setting("audio.driver", "pulseaudio")

        # Load soundfont and set up organ
        self.sfid = self.synth.sfload(self._soundfont_path)
        if self.sfid == -1:
            raise RuntimeError(f"Failed to load soundfont: {self._soundfont_path}")
        self.synth.sfont_select(0, self.sfid)
        self.synth.program_select(0, self.sfid, 0, self._instrument)

        # Start the synthesizer engine
        self.synth.start()

        # Play the initial note
        self.start()

    def start(self) -> None:
        """Start playing the continuous tone"""

        if self._is_playing:
            return  # Already playing

        self.synth.noteon(0, self._current_note, self._current_velocity)  # Start note with current velocity
        self._is_playing = True

    def stop(self) -> None:
        """Stop the tone"""
        if not self._is_playing:
            return  # Already stopped

        self.synth.noteoff(0, self._current_note)  # Stop the note
        self._is_playing = False

    def set_velocity(self, amplitude: float) -> None:
        """Set the amplitude of the tone"""

        # Convert amplitude to MIDI velocity
        velocity = int(amplitude * 127)

        # Clip velocity to valid range
        velocity = max(0, min(127, velocity))

        if velocity == self._current_velocity:
            return  # No change in amplitude

        self.synth.cc(0, 7, velocity)  # Set channel velocity
        self._current_velocity = velocity

    def cleanup(self) -> None:
        """Clean up FluidSynth resources"""
        try:
            self.stop()
            self.synth.delete()
        except Exception as e:
            print(f"Error during cleanup: {e}")

