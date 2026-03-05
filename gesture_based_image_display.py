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
#gesture detection functions
def get_finger_status(hand_landmarks):
    if not hand_landmarks:
        return [False] * 5

    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_pips = [
        mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]

    status = []
    status.append(hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x)
    for i in range(1, 5):
        status.append(hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_pips[i]].y)

    return status

def eye_aspect_ratio(eye_landmarks, face_landmarks):
    p1 = face_landmarks.landmark[eye_landmarks[0]]
    p2 = face_landmarks.landmark[eye_landmarks[1]]
    p3 = face_landmarks.landmark[eye_landmarks[2]]
    p4 = face_landmarks.landmark[eye_landmarks[3]]
    p5 = face_landmarks.landmark[eye_landmarks[4]]
    p6 = face_landmarks.landmark[eye_landmarks[5]]

    vertical_dist1 = hypot(p2.x - p6.x, p2.y - p6.y)
    vertical_dist2 = hypot(p3.x - p5.x, p3.y - p5.y)
    horizontal_dist = hypot(p1.x - p4.x, p1.y - p4.y)

    if horizontal_dist == 0:
        return 0
    return (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

def is_winking(face_landmarks):
    if not face_landmarks:
        return False

    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    ear_right = eye_aspect_ratio(RIGHT_EYE, face_landmarks)
    ear_left = eye_aspect_ratio(LEFT_EYE, face_landmarks)

    wink_ratio = 1.8
    ear_threshold = 0.22

    is_right_wink = ear_right < ear_threshold and ear_left > ear_right * wink_ratio
    is_left_wink = ear_left < ear_threshold and ear_right > ear_left * wink_ratio

    return is_right_wink or is_left_wink

def is_surprised(face_landmarks):
    if not face_landmarks:
        return False

    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    mouth_open_dist = abs(upper_lip.y - lower_lip.y)

    return mouth_open_dist > 0.03

def is_thinking(hand_landmarks, face_landmarks):
    if not hand_landmarks or not face_landmarks:
        return False

    finger_status = get_finger_status(hand_landmarks)
    thumb, index, middle, ring, pinky = finger_status
    is_pointing = not thumb and index and not middle and not ring and not pinky

    if not is_pointing:
        return False

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    lip = face_landmarks.landmark[13]  # MediaPipe landmark for the upper lip

    distance = hypot(index_tip.x - lip.x, index_tip.y - lip.y)

    proximity_threshold = 0.08
    return distance < proximity_threshold