from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, List, Optional

class ObjectDetector:
    # Known average heights of common objects in meters
    KNOWN_HEIGHTS = {
        'person': 1.7,
        'chair': 0.8,
        'car': 1.5,
        'truck': 2.5,
        'bicycle': 1.0,
        'motorcycle': 1.2,
        'dog': 0.5,
        'cat': 0.3,
    }
    
    def __init__(self, model_path: str = 'yolov8n.pt', frame_width: int = 640, frame_height: int = 320):
        """Initialize the YOLO model."""
        self.model = YOLO(model_path, task="detect")
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Camera parameters (can be calibrated)
        self.focal_length = frame_width  # A reasonable default
        self.sensor_height = frame_height
        self.vertical_fov = 50  # Typical vertical FOV for webcams (degrees)

        self.model.overrides['imgsz'] = 320
        
    def _calculate_distance(self, box_height: float, object_class: str) -> Dict[str, float]:
        """
        Calculate distance using multiple methods and return the most reliable estimate.
        
        Args:
            box_height: Height of bounding box in pixels
            object_class: Class name of detected object
            
        Returns:
            Dictionary containing distance information
        """
        # Method 1: Basic ratio method (current approach enhanced)
        ratio = box_height / self.frame_height
        basic_distance = "close" if ratio > 0.33 else "far" if ratio < 0.16 else "medium"
        
        # Method 2: Known object size method (when available)
        distance_meters = None
        confidence = "low"
        
        if object_class in self.KNOWN_HEIGHTS:
            # Using the formula: Distance = (Known Height × Focal Length) / Pixel Height
            known_height = self.KNOWN_HEIGHTS[object_class]
            distance_meters = (known_height * self.focal_length) / box_height
            
            # Adjust confidence based on object size and class
            if box_height > 20:  # Minimum size for reliable measurement
                confidence = "high" if object_class in ['person', 'car', 'truck'] else "medium"
        
        return {
            "category": basic_distance,
            "meters": distance_meters,
            "confidence": confidence
        }
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and return detection results.
        
        Args:
            frame: numpy array of the image frame
            
        Returns:
            Dictionary containing detected objects and their details
        """
        # Resize frame to specified dimensions
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Process frame with YOLO
        results = self.model(frame)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            # Calculate position in frame (left, center, right)
            center_x = (x1 + x2) / 2
            position = "left" if center_x < self.frame_width/3 else "right" if center_x > 2*self.frame_width/3 else "center"
            
            # Calculate distance using enhanced method
            box_height = y2 - y1
            distance_info = self._calculate_distance(box_height, class_name)
            
            detection = {
                "object": class_name,
                "confidence": round(confidence, 2),
                "position": position,
                "distance": distance_info["category"],
                "distance_meters": distance_info["meters"],
                "distance_confidence": distance_info["confidence"],
                "bbox": [round(x, 2) for x in [x1, y1, x2, y2]]
            }
            detections.append(detection)
            
        return {
            "detections": detections,
            "frame_size": (self.frame_height, self.frame_width)
        }

    def draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        # Resize frame if needed
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        for det in detections["detections"]:
            x1, y1, x2, y2 = map(int, det["bbox"])
            
            # Color based on distance
            color = (0, 255, 0)  # Green for far
            if det["distance"] == "close":
                color = (0, 0, 255)  # Red for close
            elif det["distance"] == "medium":
                color = (0, 165, 255)  # Orange for medium
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create detailed label with distance information
            label_parts = [det['object']]
            if det['distance_meters'] is not None:
                label_parts.append(f"{det['distance_meters']:.1f}m")
            label_parts.append(f"({det['confidence']:.2f})")
            
            label = " ".join(label_parts)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
        
    def calibrate_camera(self, known_distance: float, known_height: float, measured_pixels: float):
        """
        Calibrate camera parameters using a known object.
        
        Args:
            known_distance: Real distance to object in meters
            known_height: Real height of object in meters
            measured_pixels: Height of object in pixels
        """
        self.focal_length = (measured_pixels * known_distance) / known_height 