import cv2

def get_available_frame_sizes(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    available_sizes = []

    # Iterate through the properties of the camera
    for width in range(1, 2000, 20):
        for height in range(1, 1500, 20):
            # Set the desired width and height
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Check if the width and height were set correctly
            if cap.get(cv2.CAP_PROP_FRAME_WIDTH) == width and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == height:
                available_sizes.append((width, height))

    cap.release()

    return available_sizes

# Usage example:
camera_index = 0  # Change it if you have multiple cameras
sizes = get_available_frame_sizes(camera_index)

# Print the available frame sizes
for size in sizes:
    print(f"Width: {size[0]}, Height: {size[1]}")
