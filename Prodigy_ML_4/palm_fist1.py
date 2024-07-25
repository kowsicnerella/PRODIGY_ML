import cv2
import time
import math
import mouse
import pyautogui
import threading
import numpy as np
import mediapipe as mp
from gesture_control import control
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()

SMOOTHING_FACTOR = 0.3

INTERPOLATION_STEPS = 10

prev_x = prev_y = 0

model = load_model('gesture_recognition_model.keras')

gesture_labels = np.load('gesture_labels.npy')
label_encoder = LabelEncoder()


def control_mouse(hand_landmark, cam_frame):
    global prev_x, prev_y

    middle_finger_tip = hand_landmark.landmark[12]

    h, w, _ = cam_frame.shape
    x = float(middle_finger_tip.x) * w
    y = float(middle_finger_tip.y) * h

    n_x = 2*float(middle_finger_tip.x) - 0.5
    x = x * screen_width / w * n_x
    y = y * screen_height / h * 1.5

    smoothed_x = prev_x + SMOOTHING_FACTOR * (x - prev_x)
    smoothed_y = prev_y + SMOOTHING_FACTOR * (y - prev_y)

    min_duration, max_duration = 0.1, 0.001
    distance = math.sqrt((smoothed_x - prev_x) ** 2 + (smoothed_y - prev_y) ** 2)
    max_distance = 1920
    duration = max_duration - (max_duration - min_duration) * min(distance / max_distance, 1)

    for i in range(0, INTERPOLATION_STEPS):
        intermediate_x = prev_x + (smoothed_x - prev_x) * (i / INTERPOLATION_STEPS)
        intermediate_y = prev_y + (smoothed_y - prev_y) * (i / INTERPOLATION_STEPS)
        mouse.move(intermediate_x, intermediate_y, absolute=True, duration=duration / INTERPOLATION_STEPS)

    prev_x, prev_y = smoothed_x, smoothed_y


def is_thumb_open(thumb_hand_landmarks):
    if thumb_hand_landmarks.landmark[4].x < thumb_hand_landmarks.landmark[5].x:
        return True
    else:
        return False


def is_index_open(index_hand_landmarks):
    if index_hand_landmarks.landmark[7].y < index_hand_landmarks.landmark[8].y:
        return True
    else:
        return False


def predict_gesture(predict_landmarks):
    predict_landmarks = np.array(predict_landmarks).reshape(1, -1)
    prediction = model.predict(predict_landmarks)
    gesture = label_encoder.inverse_transform([np.argmax(prediction)])
    return gesture[0]


def is_drag_click(click_landmarks):
    x1 = click_landmarks.landmark[4].x
    x2 = click_landmarks.landmark[8].x
    y1 = click_landmarks.landmark[4].y
    y2 = click_landmarks.landmark[8].y
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    x = True if distance < 0.065 else False
    return x


gesture_labels_encoded = label_encoder.fit_transform(gesture_labels)

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev = ""
count = 0
timer = 0
index_flag = False
thumb_flag = False
drag_flag = False
mouse_drag_flag = False
thumb_flag_prev = False
mouse_operation_flag = False

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if mouse_operation_flag:
            prev_x, prev_y = mouse.get_position()

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx1, hand_landmarks1 in enumerate(results.multi_hand_landmarks):
                    if idx1 == 0:

                        landmark1 = [[lm.x, lm.y] for lm in hand_landmarks1.landmark]
                        Result = predict_gesture(np.array(landmark1))

                        if Result != "mouse":
                            mouse_operation_flag = False
                            break

                        thread = threading.Thread(target=control_mouse, args=(hand_landmarks1, frame))
                        thread.start()
                        thread.join()

                        if is_drag_click(hand_landmarks1):
                            if not drag_flag:
                                mouse.press()
                                mouse_drag_flag = True
                            drag_flag = True

                        elif is_thumb_open(hand_landmarks1):
                            if not thumb_flag:
                                timer = time.time()
                                mouse.click()
                            elif thumb_flag_prev != thumb_flag and time.time() - timer > 0.8:
                                mouse.double_click()
                                thumb_flag_prev = True
                            thumb_flag = True

                        elif is_index_open(hand_landmarks1):
                            if not index_flag:
                                mouse.click(button='right')
                                index_flag = True
                            else:
                                index_flag = False

                        else:
                            if mouse_drag_flag:
                                mouse.release()
                            timer = 0
                            thumb_flag = False
                            drag_flag = False
                            thumb_flag_prev = False

        elif results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                result = predict_gesture(np.array(landmarks))

                prev = control(prev, result)
                if prev == "mouse":
                    mouse_operation_flag = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()

cv2.destroyAllWindows()
