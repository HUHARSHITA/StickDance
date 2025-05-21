import cv2
import mediapipe as mp
import numpy as np

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Pose landmark connections for drawing the stick figure
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Colors for the stick figure
STICK_FIGURE_COLOR = (0, 255, 0)  # Green
LANDMARK_COLOR = (255, 0, 0)  # Blue

# Webcam setup
cap = cv2.VideoCapture(0)  # 0 for default webcam

#  Error handling for webcam (added robustness)
if not cap.isOpened():
    print("ERROR: Could not open video device.")
    exit()

#  Camera resolution (important for consistent performance, consider making this configurable)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# Function to draw a stick figure based on pose landmarks
def draw_stick_figure(image, landmarks, image_width, image_height):
    """Draws a stick figure on the image using the detected landmarks.

    Args:
        image: The image to draw on.
        landmarks: A list of landmarks detected by MediaPipe.
        image_width: The width of the image.
        image_height: The height of the image.
    """
    if landmarks:
        for connection in POSE_CONNECTIONS:
            start_point = landmarks[connection[0]]  # Access the element directly
            end_point = landmarks[connection[1]]    # Access the element directly

            # Check if the landmark is visible (important for robustness)
            if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                x1, y1 = int(start_point.x * image_width), int(start_point.y * image_height)
                x2, y2 = int(end_point.x * image_width), int(end_point.y * image_height)
                cv2.line(image, (x1, y1), (x2, y2), STICK_FIGURE_COLOR, 2)


# Function to create a background (optional)
def create_background(width, height):
    """Creates a simple background image.  You can customize this.

    Args:
        width: Width of the background.
        height: Height of the background.

    Returns:
        A NumPy array representing the background image.
    """
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:] = (50, 50, 50)  # Dark gray background
    return background


# MediaPipe Pose setup
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape

        # Create a blank image for the stick figure
        stick_figure_image = create_background(image_width, image_height)

        # Overlay the stick figure on the blank image
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            draw_stick_figure(stick_figure_image, landmarks, image_width, image_height)

        # Mirror the stick figure horizontally
        stick_figure_image = cv2.flip(stick_figure_image, 1)  # Flip horizontally

        # Concatenate the original image and the stick figure image side by side
        final_image = np.concatenate((image, stick_figure_image), axis=1)

        # Display the image with the stick figure
        cv2.imshow('MediaPipe Pose', final_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()