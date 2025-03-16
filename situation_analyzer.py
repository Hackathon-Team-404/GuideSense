import os
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv

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
                if has_close_obstacles and has_center_obstacles:
                    guidance = "Stop"
                    priority = "high"
                    situation = "blocked"
                elif has_close_obstacles:
                    if any(det["position"] == "left" for det in relevant_detections):
                        guidance = "Right"
                        situation = "obstacle_left"
                    else:
                        guidance = "Left"
                        situation = "obstacle_right"
                    priority = "high"
                else:
                    guidance = "Proceed"
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