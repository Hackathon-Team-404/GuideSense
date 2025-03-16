import threading
import time
import wave
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import os


class VoiceController:
    def __init__(self, activation_phrase="go"):
        """Initialize voice controller with activation phrase."""
        self.activation_phrase = activation_phrase.lower()
        self.is_listening = False
        self.is_running = True
        self.system_active = False

        # Initialize audio parameters
        self.chunk = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 3

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Initialize Whisper model (small model for good balance of speed and accuracy)
        print("Loading Whisper model... This might take a few seconds...")
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        print("Whisper model loaded!")

        # Start listening thread
        self.listen_thread = threading.Thread(
            target=self._listen_for_activation)
        self.listen_thread.daemon = True
        self.listen_thread.start()

    def _record_audio(self):
        """Record audio from microphone."""
        frames = []

        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.float32))

        stream.stop_stream()
        stream.close()

        return np.concatenate(frames)

    def _transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper."""
        # Save audio to temporary file
        temp_file = None
        wf = None
        try:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False)
            temp_file_path = temp_file.name
            temp_file.close()  # Close the file explicitly

            # Now write to the file
            wf = wave.open(temp_file_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(audio_data.tobytes())
            wf.close()  # Close the wave file explicitly

            # Transcribe using Whisper
            segments, _ = self.model.transcribe(temp_file_path, language="en")
            text = " ".join([segment.text for segment in segments]).lower()

            return text
        except Exception as e:
            print(f"Exception during transcription: {e}")
            return ""
        finally:
            # Clean up resources in finally block
            try:
                # The wave module's Wave_write object doesn't have a 'closed' attribute
                # so we need to use a different approach for cleanup
                try:
                    if wf:
                        wf.close()
                except:
                    pass  # Already closed or invalid

                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception as e:
                print(f"Cleanup error (non-critical): {e}")

    def _listen_for_activation(self):
        """Continuously listen for the activation phrase."""
        while self.is_running:
            if not self.is_listening:
                try:
                    # Record audio
                    audio_data = self._record_audio()

                    # Transcribe audio
                    text = self._transcribe_audio(audio_data)

                    if self.activation_phrase in text:
                        print(
                            "Activation phrase detected! Starting navigation assistance...")
                        self.system_active = True

                except Exception as e:
                    print(f"Error in voice recognition: {str(e)}")
                    pass  # Continue listening even if there's an error

            time.sleep(0.1)  # Short sleep to prevent CPU overuse

    def is_system_active(self) -> bool:
        """Check if the system has been activated."""
        return self.system_active

    def reset_activation(self):
        """Reset the activation state."""
        self.system_active = False

    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        if self.listen_thread.is_alive():
            # Add timeout to prevent hanging
            self.listen_thread.join(timeout=1)
        self.audio.terminate()
