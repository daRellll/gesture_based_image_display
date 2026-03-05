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

    right_eye = [33, 160, 158, 133, 153, 144]
    left_eye = [362, 385, 387, 263, 373, 380]
    ear_right = eye_aspect_ratio(right_eye, face_landmarks)
    ear_left = eye_aspect_ratio(left_eye, face_landmarks)

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

def main():
    cap = cv2.VideoCapture(0)
    current_gesture = "neutral"

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        hand_results = hands.process(image)
        face_results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detected_gesture = "neutral"

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        if is_thinking(hand_landmarks, face_landmarks):
                            detected_gesture = "think"

                if detected_gesture == "neutral":
                    finger_status = get_finger_status(hand_landmarks)
                    thumb, index, middle, ring, pinky = finger_status
                    if not thumb and index and not middle and not ring and not pinky:
                        detected_gesture = "idea"
                    elif not thumb and not index and middle and not ring and not pinky:
                        detected_gesture = "middle_finger"
                    elif thumb and not index and not middle and not ring and pinky:
                        detected_gesture = "phone"

        if detected_gesture == "neutral" and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                if is_winking(face_landmarks):
                    detected_gesture = "wink"
                elif is_surprised(face_landmarks):
                    detected_gesture = "surprised"

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.hand_connections)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Full face mesh in cyan
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.facemesh_tesselation,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
                )
                # Face outline in white
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        current_gesture = detected_gesture
