import cv2
import csv
import os

# Variable to track if drawing is in progress
drawing = False

# Variables to store rectangle coordinates
rect_start = (0, 0)
rect_end = (0, 0)

global recorded

# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    global rect_start, rect_end, drawing, picture_copy

    picture_copy = picture.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(picture_copy, rect_start, (x, y), (0, 255, 0), 2)
            cv2.imshow("Captured Image", picture_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_end = (x, y)

        # Draw the final rectangle on the original frame
        cv2.rectangle(picture_copy, rect_start, rect_end, (0, 255, 0), 2)
        cv2.imshow("Captured Image", picture_copy)

        # Call the log_object() function to save the object data
        log_object(image_file, rect_start, rect_end)

# Function to log the object data
def log_object(image_file, rect_start, rect_end):
    # Save the rectangle coordinates and image location to a CSV file
    csv_file = "object_data.csv"
    csv_header = ["Image File", "Top Left X", "Top Left Y", "Bottom Right X", "Bottom Right Y"]
    csv_row = [image_file, rect_start[0], rect_start[1], rect_end[0], rect_end[1]]

    global recorded

    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        if not file_exists:
            csv_writer.writerow(csv_header)
        csv_writer.writerow(csv_row)

    # Save the captured image to a file
    cv2.imwrite(image_file, picture)

    recorded = True

def open_camera(camera_index=0):
    # Initialize the webcam
    cap = cv2.VideoCapture(camera_index)  # 0 represents the default webcam, change it if you have multiple cameras

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Failed to open the webcam")
        exit()

    return cap

def take_picture(camera):
    # Read a frame from the webcam
    ret, frame = camera.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Failed to capture frame")
        exit()

    # Display the captured image
    cv2.imshow("Captured Image", frame)

    return frame

# Usage example:
folder = "images/"

def get_next_file_name(folder):
    file_numbers = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_prefix = 'captured_image_'
            if filename.startswith(file_prefix):
                number_str = os.path.splitext(filename[len(file_prefix):])[0]
                file_numbers.append(int(number_str))

    if file_numbers:
        next_number = max(file_numbers) + 1
    else:
        next_number = 1

    return "{}captured_image_{}.jpg".format(folder, next_number)

if __name__ == '__main__':
    camera = open_camera()

    while True:
        picture = take_picture(camera)
        image_file = get_next_file_name(folder)
        cv2.setMouseCallback("Captured Image", mouse_callback)

        recorded = False

        key = cv2.waitKey(0)

        if key == ord('q'):  # Press 'q' to quit the loop and exit the program
            break
        elif key == 32: # Press 'space' to go to next picture
            if recorded == False:
                log_object(image_file,(-1,-1),(-1,-1))
                #print("Recorded")
            else:
                #print("Not Recorded")
                pass

    # Release the webcam
    camera.release()
    cv2.destroyAllWindows()