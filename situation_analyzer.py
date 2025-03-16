import os
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv
import time
import queue
import threading
import cv2

load_dotenv()

class SituationAnalyzer:
    def __init__(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.last_analysis = None
        
    def analyze_situation(self, detection_data: Dict) -> Dict:
        """
        Analyze the detected objects and provide navigation guidance.
        Always provide feedback, even when no obstacles are detected.
        
        Args:
            detection_data: Dictionary containing detected objects and their details
            
        Returns:
            Dictionary containing analysis and recommendations
        """
        # Check for detections
        detections = detection_data.get("detections", [])
        
        # If no detections or all objects are far away, return safe status
        if not detections or all(det["distance"] == "far" for det in detections):
            analysis = {
                "safe_to_proceed": True,
                "guidance": "Clear",
                "priority": "low",
                "situation": "no_obstacles"
            }
        else:
            # Filter out far objects and low confidence detections
            relevant_detections = [
                det for det in detections 
                if det["distance"] != "far" and det["confidence"] > 0.3
            ]
            
            if not relevant_detections:
                analysis = {
                    "safe_to_proceed": True,
                    "guidance": "Clear",
                    "priority": "low",
                    "situation": "no_obstacles"
                }
            else:
                # Analyze the situation based on object positions
                has_close_obstacles = any(det["distance"] == "close" for det in relevant_detections)
                has_center_obstacles = any(det["position"] == "center" and det["distance"] != "far" 
                                        for det in relevant_detections)
                
                # Generate concise guidance based on situation
                guidance_parts = []
                for det in relevant_detections:
                    distance_meters = det.get("distance_meters", None)
                    if distance_meters is not None:
                        guidance_parts.append(f"{det['object']} {distance_meters:.1f}m")
                
                guidance = ", ".join(guidance_parts)
                
                if has_close_obstacles and has_center_obstacles:
                    guidance = "Stop! " + guidance
                    priority = "high"
                    situation = "blocked"
                elif has_close_obstacles:
                    if any(det["position"] == "left" for det in relevant_detections):
                        guidance = "Left: " + guidance
                        situation = "obstacle_left"
                    else:
                        guidance = "Right: " + guidance
                        situation = "obstacle_right"
                    priority = "high"
                else:
                    guidance = "Clear: " + guidance
                    priority = "low"
                    situation = "path_clear"
                
                analysis = {
                    "safe_to_proceed": not (has_close_obstacles and has_center_obstacles),
                    "guidance": guidance,
                    "priority": priority,
                    "situation": situation
                }
        
        # Check if the situation has changed
        if self.last_analysis and self.last_analysis["situation"] == analysis["situation"]:
            analysis["changed"] = False
        else:
            analysis["changed"] = True
            self.last_analysis = analysis.copy()
        
        return analysis
        
    def _create_situation_description(self, detections: List[Dict]) -> str:
        """Create a natural language description of the detected objects."""
        descriptions = []
        for det in detections:
            desc = (f"A {det['object']} is {det['distance']} away in the {det['position']} "
                   f"position with {det['confidence']*100:.0f}% confidence")
            descriptions.append(desc)
        
        return " ".join(descriptions)

class AudioFeedback:
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
            
            # Immediately stop current audio and play new message
            self.engine.stop()  # Stop any ongoing speech
            
            # Clear old messages if queue is full
            while self.message_queue.full():
                try:
                    self.message_queue.get_nowait()
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
            
        except Exception as e:
            print(f"Error queueing feedback: {e}") 

class VoiceController:
    def __init__(self, activation_phrase="go", stop_phrase="i'm here"):
        """Initialize voice controller with activation and stop phrases."""
        self.activation_phrase = activation_phrase.lower()
        self.stop_phrase = stop_phrase.lower()
        self.is_listening = False
        self.is_running = True
        self.system_active = False
        self.should_stop = False  # New attribute to track stop command
        
        # Initialize audio parameters
        # ... existing code ...
        
        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_for_activation)
        self.listen_thread.daemon = True
        self.listen_thread.start()
    
    def _listen_for_activation(self):
        """Continuously listen for the activation and stop phrases."""
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
                    elif self.stop_phrase in text:
                        print("Stop phrase detected! Stopping application...")
                        self.should_stop = True
                        self.is_running = False  # Stop the listening loop
                        
                except Exception as e:
                    print(f"Error in voice recognition: {str(e)}")
                    pass  # Continue listening even if there's an error
                    
            time.sleep(0.1)  # Short sleep to prevent CPU overuse
    
    def should_stop_application(self) -> bool:
        """Check if the stop command has been issued."""
        return self.should_stop 