import cv2
import time
from object_detector import ObjectDetector
from situation_analyzer import SituationAnalyzer
from audio_feedback import AudioFeedback
from voice_control import VoiceController
import pyttsx3
import threading
import queue

def main():
    # Initialize components
    detector = ObjectDetector(model_path="yolov8n.pt", frame_width=640, frame_height=320)
    analyzer = SituationAnalyzer()
    audio = AudioFeedback()
    voice_control = VoiceController()
    
    # Initialize video capture (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        
    print("Waiting for activation phrase: 'Go'...")
    print("Press 'q' to quit")
    
    last_analysis_time = 0
    analysis_interval = 1.0  # Analysis every 1 second
    last_detection_results = None
    running = True
    
    try:
        while running:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                continuepy
            
            # Check if system is activated
            if voice_control.is_system_active():
                current_time = time.time()
                
                # Only process frame at the specified interval
                if current_time - last_analysis_time >= analysis_interval:
                    # Process frame with YOLO
                    detection_results = detector.process_frame(frame)
                    
                    # Analyze situation only if detections have changed
                    if _has_detections_changed(last_detection_results, detection_results):
                        # Analyze situation
                        analysis = analyzer.analyze_situation(detection_results)
                        
                        # Only provide audio feedback if the situation has changed
                        if analysis.get("changed", True):
                            audio.provide_feedback(analysis)
                        
                        last_detection_results = detection_results
                    
                    last_analysis_time = current_time
                    
                    # Get audio feedback stats
                    stats = audio.get_stats()
                    
                    # Add visual status indicators
                    status_color = (0, 255, 0) if analysis.get("safe_to_proceed", True) else (0, 0, 255)
                    cv2.putText(frame, f"ACTIVE - {analysis.get('guidance', '')}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    # Display audio feedback stats
                    delay_color = (0, 255, 0) if stats['last_message_delay'] < 0.5 else (0, 165, 255)
                    cv2.putText(frame, f"Audio Delay: {stats['last_message_delay']:.2f}s", 
                               (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, delay_color, 2)
                    cv2.putText(frame, f"Queue Size: {stats['queue_size']}", 
                               (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Always draw detection boxes on frame
                if last_detection_results:
                    frame = detector.draw_detections(frame, last_detection_results)
            else:
                # Draw waiting message on frame
                cv2.putText(frame, "Waiting for activation: 'Go'", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show processed frame
            cv2.imshow('Wheelchair Navigation Assistant', frame)
            
            # Check for quit command
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                print("\nQuitting application...")
                
    except KeyboardInterrupt:
        print("\nStopping application...")
    finally:
        # Cleanup
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        audio.cleanup()
        voice_control.cleanup()

def _has_detections_changed(last_results: dict, current_results: dict) -> bool:
    """Compare detection results to determine if there's a significant change."""
    if last_results is None:
        return True
        
    last_detections = last_results.get("detections", [])
    current_detections = current_results.get("detections", [])
    
    # Compare number of detections
    if len(last_detections) != len(current_detections):
        return True
    
    # Create simplified representations for comparison
    def simplify_detection(det):
        return (
            det["object"],
            det["distance"],
            det["position"]
        )
    
    last_simplified = {simplify_detection(det) for det in last_detections}
    current_simplified = {simplify_detection(det) for det in current_detections}
    
    return last_simplified != current_simplified

if __name__ == "__main__":
    main() 