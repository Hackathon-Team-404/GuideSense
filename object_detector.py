from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, List

class ObjectDetector:
    def __init__(self, model_path: str = 'yolov8n.pt', frame_width: int = 640, frame_height: int = 320):
        """Initialize the YOLO model."""
        self.model = YOLO(model_path)
        self.frame_width = frame_width
        self.frame_height = frame_height
        
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
            
            # Calculate approximate distance based on box size
            box_height = y2 - y1
            distance = "close" if box_height > self.frame_height/3 else "far" if box_height < self.frame_height/6 else "medium"
            
            detection = {
                "object": class_name,
                "confidence": round(confidence, 2),
                "position": position,
                "distance": distance,
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['object']} ({det['confidence']:.2f})"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame 