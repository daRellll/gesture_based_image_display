import cv2
import mediapipe as mp
import numpy as np
import os
from math import hypot

#mediapipe hands and face detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# Using a thinner drawing spec for a cleaner look on the face mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# gesture to image
image_dir = "gesture_images"

#gesture names and corresponding images
gesture_map = {
    "idea": "idea.jpg",
    "middle_finger": "fyou.jpg",
    "phone": "phone.jpg",
    "wink": "wink.jpg",
    "surprised": "shocked.jpg",
    "think": "think.jpg",
    "neutral": "idle.jpg",
}

#image loader
gesture_images = {}
for gesture, filename in gesture_map.items():
    path = os.path.join(image_dir, filename)
    image = cv2.imread(path)
    if image is None:
        print(f"Warning: Could not load image for '{gesture}' at '{path}'. Creating a placeholder.")
        image = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(image, gesture, (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    gesture_images[gesture] = image
