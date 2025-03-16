import subprocess
from typing import Dict
import threading
import queue
import time
import platform


class AudioFeedback:
    def __init__(self):
        """Initialize audio feedback system."""
        self.message_queue = queue.Queue(maxsize=3)  # Increased queue size
        self.is_running = True
        self.speech_thread = threading.Thread(
            target=self._process_speech_queue)
        self.speech_thread.daemon = True

        # Message deduplication
        self.last_message = ""
        self.last_message_time = 0
        self.message_cooldown = 0.5  # Reduced cooldown to allow more frequent updates

        # Debug info
        self.message_count = 0
        self.last_processed_time = 0
        self.processing_delay = 0

        # Check if we're on macOS
        self.use_say = platform.system() == 'Darwin'

        self.speech_thread.start()

    def provide_feedback(self, analysis: Dict):
        """Queue feedback message for speech output."""
        if not self.is_running:
            return

        try:
            priority = analysis.get("priority", "low")
            guidance = analysis.get("guidance", "")

            # Skip empty guidance
            if not guidance:
                return

            current_time = time.time()

            # Only skip identical messages that occur within cooldown time
            skip_message = False
            if guidance == self.last_message and current_time - self.last_message_time < self.message_cooldown:
                skip_message = True

            # For debugging - print when messages are skipped
            if skip_message:
                print(
                    f"Skipping duplicate message: '{guidance}' (last sent {current_time - self.last_message_time:.2f}s ago)")
                return

            # Clear old messages if queue is full - but keep high priority messages
            while self.message_queue.full():
                try:
                    old_message = self.message_queue.get_nowait()
                    # If this is high priority and an old message is low priority, put high priority back
                    if priority == "high" and old_message.get('priority') == "low":
                        continue
                    self.message_queue.task_done()
                except queue.Empty:
                    break

            self.last_message = guidance
            self.last_message_time = current_time

            if priority == "high":
                guidance = "Alert! " + guidance

            # Add message with timestamp
            self.message_queue.put({
                'text': guidance,
                'timestamp': current_time,
                'priority': priority
            })

            self.message_count += 1
            print(
                f"Queued message: '{guidance}' (queue size: {self.message_queue.qsize()})")

        except Exception as e:
            print(f"Error queueing feedback: {e}")

    def _speak_message(self, message_data: Dict):
        """Speak a message using the appropriate method."""
        try:
            text = message_data['text']
            delay = time.time() - message_data['timestamp']
            self.processing_delay = delay

            # Only skip old messages if they are low priority and very old (increased from 1.5s to 3.0s)
            if delay > 3.0 and message_data['priority'] != 'high':
                print(f"Skipping old message '{text}' (delay: {delay:.2f}s)")
                return

            # Print when we're actually speaking a message
            print(f"Speaking: '{text}' (delay: {delay:.2f}s)")

            if self.use_say:
                # Use macOS 'say' command with faster rate for better real-time performance
                # Increased speech rate from 180 to 220
                subprocess.run(['say', '-r', '220', text], check=True)
            else:
                # Fallback to pyttsx3 for other platforms
                import pyttsx3
                engine = pyttsx3.init()
                # Even faster rate for short commands
                # Increased speech rate from 180 to 220
                engine.setProperty('rate', 220)
                engine.say(text)
                engine.runAndWait()
                engine.stop()

            self.last_processed_time = time.time()

        except Exception as e:
            print(f"Speech error (skipping message): {e}")

    def _process_speech_queue(self):
        """Process queued speech messages in a separate thread."""
        while self.is_running:
            try:
                # Get message with timeout to allow checking is_running
                message_data = self.message_queue.get(timeout=0.5)

                if not self.is_running:
                    break

                self._speak_message(message_data)
                self.message_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech processing: {e}")

    def get_stats(self) -> Dict:
        """Get current audio feedback statistics."""
        return {
            'queue_size': self.message_queue.qsize(),
            'messages_processed': self.message_count,
            'last_message_delay': self.processing_delay,
            'last_processed': time.time() - self.last_processed_time if self.last_processed_time else 0
        }

    def cleanup(self):
        """Clean up resources."""
        self.is_running = False

        # Clear message queue
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except queue.Empty:
                break

        # Stop speech thread
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1)
