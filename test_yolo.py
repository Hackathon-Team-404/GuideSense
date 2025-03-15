from ultralytics import YOLO

# Load a YOLO v8 model pretrained on the COCO dataset
model = YOLO('yolov8n.pt')

# Run detection on an image file
results = model('input.jpg')

# Visualize the detections; this opens a window with the image showing bounding boxes
results[0].show()


for result in results:
    # result.boxes contains the bounding box data
    print(result.boxes.data)