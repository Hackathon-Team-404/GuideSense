import os
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

class SituationAnalyzer:
    def __init__(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
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
        detections = detection_data["detections"]
        
        # If no detections, return safe status
        if not detections:
            return {
                "safe_to_proceed": True,
                "guidance": "Path is clear and safe.",
                "priority": "low"
            }
            
        # Create a detailed situation description
        situation = self._create_situation_description(detections)
        
        # Query the LLM
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a navigation assistant for a wheelchair user. 
                Analyze the detected objects and their positions to provide clear, concise guidance.
                Focus on safety and obstacle avoidance. Keep responses brief and actionable.
                If the path is clear or objects are far away, indicate it's safe to proceed."""},
                {"role": "user", "content": f"Based on these detected objects: {situation}, provide navigation guidance."}
            ],
            max_tokens=150
        )
        
        # Process the response
        guidance = response.choices[0].message.content.strip()
        
        # Determine if it's safe to proceed based on the guidance and object positions
        has_close_obstacles = any(det["distance"] == "close" for det in detections)
        has_warning_words = any(word in guidance.lower() 
                              for word in ["stop", "danger", "warning", "caution", "halt"])
        
        safe_to_proceed = not (has_close_obstacles or has_warning_words)
        
        # Determine priority level
        priority = "high" if not safe_to_proceed else "low"
        
        return {
            "safe_to_proceed": safe_to_proceed,
            "guidance": guidance,
            "priority": priority
        }
        
    def _create_situation_description(self, detections: List[Dict]) -> str:
        """Create a natural language description of the detected objects."""
        descriptions = []
        for det in detections:
            desc = (f"A {det['object']} is {det['distance']} away in the {det['position']} "
                   f"position with {det['confidence']*100:.0f}% confidence")
            descriptions.append(desc)
        
        return " ".join(descriptions) 