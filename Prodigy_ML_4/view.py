import math

import cv2
import numpy as np
import mediapipe as mp

# Load the .npy file
labels = np.load("gesture_labels.npy")
data = np.load("gesture_data.npy")

# Print the data (might be truncated for large arrays)
print(data[0])

# Check the data shape and type
print(data.shape)
print(labels.shape)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands

with mp_hands.Hands() as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # print(frame.shape)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the MediaPipe Hands model.
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            x1 = results.multi_hand_landmarks[0].landmark[4].x
            x2 = results.multi_hand_landmarks[0].landmark[8].x
            y1 = results.multi_hand_landmarks[0].landmark[4].y
            y2 = results.multi_hand_landmarks[0].landmark[8].y
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance<0.08:
                print(distance)

        cv2.imshow('Palm_Fist Gesture Tracking', frame)

        # Press 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
