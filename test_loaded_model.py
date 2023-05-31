import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.saved_model.load('model/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model/')  # Or load the frozen graph

# Set up the webcam
cap = cv2.VideoCapture(0)  # 0 represents the default webcam, change it if you have multiple cameras

# Loop to capture and process frames from the webcam
while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Preprocess the frame
    processed_frame = cv2.resize(frame, (640, 640))
    #processed_frame = processed_frame / 255.0  # Normalize pixel values

    # Perform object detection
    input_tensor = np.expand_dims(processed_frame, axis=0)  # Add batch dimension
    detections = model(input_tensor)  # Perform object detection on the frame

    # Process the detection results
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    # Filter detections based on confidence threshold
    threshold = 0.7
    filtered_boxes = boxes[scores > threshold]
    filtered_classes = classes[scores > threshold]

    # Visualize the results
    for box, cls in zip(filtered_boxes, filtered_classes):
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * frame.shape[1])
        xmax = int(xmax * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        ymax = int(ymax * frame.shape[0])

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Draw class label
        label = f"Class: {cls}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the output
    cv2.imshow('Object Detection', frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit the loop and exit the program
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
