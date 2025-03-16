import os
import base64
import cv2
import io
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class SituationAnalyzer:
    def __init__(self):
        """Initialize OpenAI and Grok API clients."""
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Initialize Grok client with XAI API key
        self.grok_client = OpenAI(
            api_key=os.getenv('XAI_API_KEY'),
            base_url="https://api.x.ai/v1",
        )

        self.last_analysis = None
        self.use_grok = True  # Flag to enable/disable Grok descriptions
        self.grok_model = "grok-2-vision-latest"  # Correct model name for Grok vision

    def analyze_situation(self, detection_data: Dict, frame=None) -> Dict:
        """
        Analyze the detected objects and provide navigation guidance.
        Always provide feedback, even when no obstacles are detected.

        Args:
            detection_data: Dictionary containing detected objects and their details
            frame: Optional raw camera frame for vision analysis

        Returns:
            Dictionary containing analysis and recommendations
        """
        # Check for detections
        detections = detection_data.get("detections", [])

        # Basic analysis (similar to original implementation)
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
                has_close_obstacles = any(
                    det["distance"] == "close" for det in relevant_detections)
                has_center_obstacles = any(det["position"] == "center" and det["distance"] != "far"
                                           for det in relevant_detections)

                # Generate concise guidance based on situation
                if has_close_obstacles and has_center_obstacles:
                    guidance = "Stop"
                    priority = "high"
                    situation = "blocked"
                elif has_close_obstacles:
                    if any(det["position"] == "left" and det["distance"] == "close" for det in relevant_detections):
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

        # Add detailed scene description if enabled
        if self.use_grok:
            try:
                # If we have a frame, use Grok Vision to analyze it directly
                if frame is not None:
                    scene_description = self._analyze_image_with_grok(
                        frame, detections)
                else:
                    # Fall back to text-based description
                    scene_description = self._create_situation_description(
                        detections)

                analysis["detailed_description"] = scene_description
            except Exception as e:
                print(f"Error with Grok description: {e}")
                analysis["detailed_description"] = self._create_situation_description(
                    detections)
        else:
            analysis["detailed_description"] = self._create_situation_description(
                detections)

        return analysis

    def _create_situation_description(self, detections: List[Dict]) -> str:
        """Create a natural language description of the detected objects."""
        descriptions = []
        for det in detections:
            desc = (f"A {det['object']} is {det['distance']} away in the {det['position']} "
                    f"position with {det['confidence']*100:.0f}% confidence")
            descriptions.append(desc)

        return " ".join(descriptions)

    def _analyze_image_with_grok(self, frame, detections: List[Dict]) -> str:
        """
        Use Grok Vision API to analyze the camera frame directly.

        Args:
            frame: Raw camera frame (numpy array)
            detections: List of detection dictionaries (for context)

        Returns:
            Detailed description of the scene from Grok
        """
        try:
            # Convert the frame to base64 encoded string
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')

            # Create a prompt that includes context from YOLO detections
            detection_context = self._create_situation_summary(detections)
            prompt = (f"You are a navigation assistant for a wheelchair user. "
                      f"Describe this scene in 1-2 sentences, focusing on navigation hazards and paths. "
                      f"Context from object detection: {detection_context}")

            # Create messages for Grok Vision API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            # Call Grok Vision API
            completion = self.grok_client.chat.completions.create(
                model=self.grok_model,
                messages=messages,
                temperature=0.2,
                max_tokens=100  # Keep descriptions brief
            )

            # Extract the description from the response
            description = completion.choices[0].message.content.strip()
            return description

        except Exception as e:
            print(f"Grok Vision API error: {e}")
            # Fall back to basic description
            return self._create_situation_description(detections)

    def _create_situation_summary(self, detections: List[Dict]) -> str:
        """Create a concise summary of detections for the prompt."""
        if not detections:
            return "No objects detected"

        # Create a summary of object counts by type and position
        object_counts = {}
        for det in detections:
            key = (det['object'], det['position'], det['distance'])
            object_counts[key] = object_counts.get(key, 0) + 1

        # Create a structured scene description
        scene_elements = []
        for (obj_type, position, distance), count in object_counts.items():
            count_str = f"{count} " if count > 1 else ""
            scene_elements.append(
                f"{count_str}{obj_type}(s) {distance} away to the {position}")

        return ", ".join(scene_elements)

    def toggle_grok_descriptions(self, enabled: bool = True):
        """Enable or disable Grok descriptions."""
        self.use_grok = enabled
        return {"status": "enabled" if enabled else "disabled"}
