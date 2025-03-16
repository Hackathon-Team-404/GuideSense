import threading
import time
import wave
import pyaudio
import numpy as np
import onnxruntime
import tempfile
import os
# from pynput import keyboard


class ONNXEncoderWrapper:
    def __init__(self, encoder_path):
        """Initialize ONNX encoder with QNN execution provider."""
        self.session = self._get_onnxruntime_session(encoder_path)

    def _get_onnxruntime_session(self, path):
        """Create an ONNX runtime session with QNN provider."""
        print("Available providers:", onnxruntime.get_available_providers())
        options = onnxruntime.SessionOptions()
        try:
            session = onnxruntime.InferenceSession(
                path,
                sess_options=options,
                providers=["QNNExecutionProvider"],
                provider_options=[
                    {
                        "backend_path": "QnnHtp.dll",
                        "htp_performance_mode": "burst",
                        "high_power_saver": "sustained_high_performance",
                        "enable_htp_fp16_precision": "1",
                        "htp_graph_finalization_optimization_mode": "3",
                    }
                ],
            )
            print("Using QNN provider for encoder")
        except Exception as e:
            print(f"QNN provider failed for encoder: {e}, falling back to CPU")
            session = onnxruntime.InferenceSession(
                path,
                providers=["CPUExecutionProvider"],
            )
            print("Using CPU provider for encoder")
        
        print("Session providers:", session.get_providers())
        return session

    def to(self, *args):
        """Dummy method to match PyTorch API."""
        return self

    def __call__(self, audio):
        """Run the encoder on audio input."""
        return self.session.run(None, {"audio": audio})


class ONNXDecoderWrapper:
    def __init__(self, decoder_path):
        """Initialize ONNX decoder with QNN execution provider."""
        self.session = self._get_onnxruntime_session(decoder_path)

    def _get_onnxruntime_session(self, path):
        """Create an ONNX runtime session with QNN provider."""
        options = onnxruntime.SessionOptions()
        try:
            session = onnxruntime.InferenceSession(
                path,
                sess_options=options,
                providers=["QNNExecutionProvider"],
                provider_options=[
                    {
                        "backend_path": "QnnHtp.dll",
                        "htp_performance_mode": "burst",
                        "high_power_saver": "sustained_high_performance",
                        "enable_htp_fp16_precision": "1",
                        "htp_graph_finalization_optimization_mode": "3",
                    }
                ],
            )
            print("Using QNN provider for decoder")
        except Exception as e:
            print(f"QNN provider failed for decoder: {e}, falling back to CPU")
            session = onnxruntime.InferenceSession(
                path,
                providers=["CPUExecutionProvider"],
            )
            print("Using CPU provider for decoder")
        
        print("Session providers:", session.get_providers())
        return session

    def to(self, *args):
        """Dummy method to match PyTorch API."""
        return self

    def __call__(self, x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self):
        """Run the decoder with the given inputs."""
        return self.session.run(
            None,
            {
                "x": x.astype(np.int32),
                "index": np.array(index),
                "k_cache_cross": k_cache_cross,
                "v_cache_cross": v_cache_cross,
                "k_cache_self": k_cache_self,
                "v_cache_self": v_cache_self,
            },
        )


class VoiceController:
    def __init__(self, activation_phrase="go", encoder_path=None, decoder_path=None):
        """Initialize voice controller with activation phrase and ONNX model paths."""
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

        # Set default model paths if not provided
        if encoder_path is None:
            encoder_path = "whisper_base_en-whisperencoder.onnx"
        if decoder_path is None:
            decoder_path = "whisper_base_en-whisperdecoder.onnx"

        # Initialize Whisper model with ONNX runtime
        print("Loading Whisper ONNX model... This might take a few seconds...")
        try:
            # Import the necessary components from the qai_hub_models
            from qai_hub_models.models._shared.whisper.model import Whisper
            from qai_hub_models.models.whisper_base_en import App as WhisperApp

            # Create custom ONNX-based Whisper model
            class WhisperBaseEnONNX(Whisper):
                def __init__(self, encoder_wrapper, decoder_wrapper):
                    return super().__init__(
                        encoder_wrapper,
                        decoder_wrapper,
                        num_decoder_blocks=6,
                        num_heads=8,
                        attention_dim=512,
                    )

            # Initialize model wrappers
            encoder_wrapper = ONNXEncoderWrapper(encoder_path)
            decoder_wrapper = ONNXDecoderWrapper(decoder_path)
            
            # Initialize the whisper model and app
            whisper_model = WhisperBaseEnONNX(encoder_wrapper, decoder_wrapper)
            self.model = WhisperApp(whisper_model)
            
            print("Whisper ONNX model loaded successfully!")
        except Exception as e:
            print(f"Error loading Whisper ONNX model: {e}")
            print("Please ensure qai_hub_models is installed and model paths are correct.")
            raise

        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_for_activation)
        self.listen_thread.daemon = True
        self.listen_thread.start()

    #     # Start the keyboard listener thread
    #     self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
    #     self.keyboard_listener.start()

    # def _on_key_press(self, key):
    #     """Handle key press event to trigger activation."""
    #     try:
    #         if key.char == 's':  # If 's' key is pressed, activate the system
    #             print("Keyboard 's' pressed! Activating system.")
    #             self.system_active = True
    #     except AttributeError:
    #         pass  # Ignore special keys (e.g., Shift, Ctrl, etc.)

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

        print("Recording...")
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.float32))

        stream.stop_stream()
        stream.close()
        print("Recording finished")

        return np.concatenate(frames)

    def _transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper ONNX model."""
        # Save audio to temporary file
        temp_file = None
        wf = None
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file_path = temp_file.name
            temp_file.close()  # Close the file explicitly

            # Now write to the file
            wf = wave.open(temp_file_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(audio_data.tobytes())
            wf.close()  # Close the wave file explicitly

            # Transcribe using Whisper ONNX model
            print("Transcribing...")
            text = self.model.transcribe(temp_file_path)
            text = text.lower()
            print(f"Transcription: {text}")

            return text
        except Exception as e:
            print(f"Exception during transcription: {e}")
            import traceback
            traceback.print_exc()
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
                        print("Activation phrase detected! Starting navigation assistance...")
                        self.system_active = True

                except Exception as e:
                    print(f"Error in voice recognition: {str(e)}")
                    import traceback
                    traceback.print_exc()
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