import speech_recognition as sr
import threading
import time

class VoiceController:
    def __init__(self, activation_phrase="let's go"):
        """Initialize voice controller with activation phrase."""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.activation_phrase = activation_phrase.lower()
        self.is_listening = False
        self.is_running = True
        self.system_active = False
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_for_activation)
        self.listen_thread.daemon = True
        self.listen_thread.start()
    
    def _listen_for_activation(self):
        """Continuously listen for the activation phrase."""
        while self.is_running:
            if not self.is_listening:
                try:
                    with self.microphone as source:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        if self.activation_phrase in text:
                            print("Activation phrase detected! Starting navigation assistance...")
                            self.system_active = True
                    except sr.UnknownValueError:
                        pass  # Speech was unclear
                    except sr.RequestError:
                        print("Could not request results from speech recognition service")
                        
                except (sr.WaitTimeoutError, Exception) as e:
                    pass  # Timeout or other error, continue listening
                    
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
            self.listen_thread.join() 