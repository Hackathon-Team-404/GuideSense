import cv2
import time
from object_detector import ObjectDetector
from situation_analyzer import SituationAnalyzer
from audio_feedback import AudioFeedback
from voice_control import VoiceController

def main():
    # Initialize components
    detector = ObjectDetector(frame_width=640, frame_height=320)
    analyzer = SituationAnalyzer()
    audio = AudioFeedback()
    voice_control = VoiceController()
    
    # Initialize video capture (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        
    print("Waiting for activation phrase: 'Let's go'...")
    print("Press 'q' to quit")
    
    last_analysis_time = 0
    analysis_interval = 0.5  # Analyze every 0.5 seconds for more frequent updates
    running = True
    
    try:
        while running:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                continue  # Continue instead of break to keep trying
            
            # Check if system is activated
            if voice_control.is_system_active():
                # Process frame with YOLO
                detection_results = detector.process_frame(frame)
                
                # Draw detection boxes on frame
                frame = detector.draw_detections(frame, detection_results)
                
                # Analyze situation and provide audio feedback at intervals
                current_time = time.time()
                if current_time - last_analysis_time >= analysis_interval:
                    # Analyze situation
                    analysis = analyzer.analyze_situation(detection_results)
                    
                    # Provide audio feedback
                    audio.provide_feedback(analysis)
                    
                    last_analysis_time = current_time
                    
                # Add visual status indicator
                status_color = (0, 255, 0) if analysis.get("safe_to_proceed", True) else (0, 0, 255)
                cv2.putText(frame, "ACTIVE - " + analysis.get("guidance", "Processing..."), 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            else:
                # Draw waiting message on frame
                cv2.putText(frame, "Waiting for activation: 'Let's go'", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show processed frame
            cv2.imshow('Wheelchair Navigation Assistant', frame)
            
            # Check for quit command - only way to exit
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

if __name__ == "__main__":
    main() 