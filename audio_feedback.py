import pyttsx3
from typing import Dict
import threading
import queue
import time

class AudioFeedback:
    def __init__(self):
        """Initialize text-to-speech engine."""
        try:
            self.engine = None
            self._init_engine()
            self.message_queue = queue.Queue()
            self.is_running = True
            self.speech_thread = threading.Thread(target=self._process_speech_queue)
            self.speech_thread.daemon = True
            self.speech_thread.start()
            self.last_message = ""
            self.last_message_time = 0
            self.message_cooldown = 1.0  # Minimum time between identical messages
        except Exception as e:
            print(f"Error initializing audio feedback: {e}")
            raise

    def _init_engine(self):
        """Initialize or reinitialize the speech engine."""
        if self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def provide_feedback(self, analysis: Dict):
        """
        Queue feedback message for speech output.
        
        Args:
            analysis: Dictionary containing situation analysis and guidance
        """
        try:
            priority = analysis.get("priority", "low")
            guidance = analysis.get("guidance", "No guidance available")
            
            # Avoid repeating the same message too frequently
            current_time = time.time()
            if (guidance == self.last_message and 
                current_time - self.last_message_time < self.message_cooldown):
                return
            
            self.last_message = guidance
            self.last_message_time = current_time
            
            # Adjust speech properties based on priority
            if priority == "high":
                guidance = "ATTENTION! " + guidance
            
            self.message_queue.put(guidance)
        except Exception as e:
            print(f"Error queueing feedback: {e}")
        
    def _process_speech_queue(self):
        """Process queued speech messages in a separate thread."""
        while self.is_running:
            try:
                message = self.message_queue.get(timeout=1)
                try:
                    self.engine.say(message)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"Speech engine error, reinitializing: {e}")
                    self._init_engine()  # Reinitialize engine on error
                    self.engine.say(message)  # Try again
                    self.engine.runAndWait()
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech processing: {e}")
            time.sleep(0.1)  # Small delay to prevent CPU overuse
                
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1)
        if self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass 