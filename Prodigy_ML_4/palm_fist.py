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


# Assuming screen width and height are known
screen_width, screen_height = pyautogui.size()

# Smoothing factor to reduce jitter
SMOOTHING_FACTOR = 0.3

# Interpolation steps for smooth movement
INTERPOLATION_STEPS = 10

# Initialize previous positions
prev_x = prev_y = 0

model = load_model('gesture_recognition_model.keras')

gesture_labels = np.load('gesture_labels.npy')
label_encoder = LabelEncoder()


def control_mouse(hand_landmark, cam_frame):
    global prev_x, prev_y

    middle_finger_tip = hand_landmark.landmark[12]

    # Convert normalized coordinates to pixel coordinates
    h, w, _ = cam_frame.shape
    x = float(middle_finger_tip.x) * w
    y = float(middle_finger_tip.y) * h

    n_x = 2*float(middle_finger_tip.x) - 0.5

    # Normalize coordinates to screen dimensions
    x = x * screen_width / w * n_x
    y = y * screen_height / h * 1.5

    # Apply smoothing to reduce jitteriness
    smoothed_x = prev_x + SMOOTHING_FACTOR * (x - prev_x)
    smoothed_y = prev_y + SMOOTHING_FACTOR * (y - prev_y)

    # print("x, y:", smoothed_x, smoothed_y)

    # Calculate distance for movement duration
    min_duration, max_duration = 0.1, 0.001
    distance = math.sqrt((smoothed_x - prev_x) ** 2 + (smoothed_y - prev_y) ** 2)
    max_distance = 1920  # Adjust this value based on your requirements
    duration = max_duration - (max_duration - min_duration) * min(distance / max_distance, 1)

    # Interpolate between previous and current positions
    for i in range(0, INTERPOLATION_STEPS):
        intermediate_x = prev_x + (smoothed_x - prev_x) * (i / INTERPOLATION_STEPS)
        intermediate_y = prev_y + (smoothed_y - prev_y) * (i / INTERPOLATION_STEPS)
        mouse.move(intermediate_x, intermediate_y, absolute=True, duration=duration / INTERPOLATION_STEPS)

    # Update previous positions
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
# Initialize the MediaPipe Hands model and drawing utilities.

mp_hands = mp.solutions.hands
# Create a video capture object.

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
# Initialize the hands model.

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    # Loop over the video frames.
    while cap.isOpened():
        # Capture a frame.
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally to create a mirror image.
        frame = cv2.flip(frame, 1)
        # print(frame.shape)

        # Convert the frame to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the MediaPipe Hands model.
        results = hands.process(frame_rgb)

        if mouse_operation_flag:
            prev_x, prev_y = mouse.get_position()

            # Draw the hand landmarks on the frame.
            if results.multi_hand_landmarks and results.multi_handedness:
                for idx1, hand_landmarks1 in enumerate(results.multi_hand_landmarks):
                    if idx1 == 0:
                        mp_drawing.draw_landmarks(frame, hand_landmarks1, mp_hands.HAND_CONNECTIONS)

                        landmark1 = [[lm.x, lm.y] for lm in hand_landmarks1.landmark]
                        Result = predict_gesture(np.array(landmark1))
                        # Result = 'pass'
                        cv2.putText(frame, Result, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3,
                                    cv2.LINE_AA)
                        if Result != "mouse":
                            print("Exit mouse")
                            mouse_operation_flag = False
                            break

                        thread = threading.Thread(target=control_mouse, args=(hand_landmarks1, frame))
                        thread.start()
                        thread.join()

                        if is_drag_click(hand_landmarks1):
                            if not drag_flag:
                                mouse.press()
                                print("drag")
                                mouse_drag_flag = True
                            drag_flag = True

                        elif is_thumb_open(hand_landmarks1):
                            print("True")
                            if not thumb_flag:
                                timer = time.time()
                                mouse.click()
                            elif thumb_flag_prev != thumb_flag and time.time() - timer > 0.8:
                                mouse.double_click()
                                print("double clicked")
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
                            print("False")
                            timer = 0
                            thumb_flag = False
                            drag_flag = False
                            thumb_flag_prev = False

        # Draw the hand landmarks on the frame.
        elif results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                # Get the hand label (left or right)
                hand_label = results.multi_handedness[idx].classification[0].label

                # print("hand: ", idx, results.multi_hand_landmarks, results.multi_handedness)

                # Draw the landmarks on the frame.
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

                result = predict_gesture(np.array(landmarks))
                print(result)

                # Display the recognized number on the frame
                cv2.putText(frame, result, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3,
                            cv2.LINE_AA)

                prev = control(prev, result)
                if prev == "mouse":
                    print("Enter mouse")
                    mouse_operation_flag = True  # Set flag to True

        # Display the frame.
        cv2.imshow('Palm_Fist Gesture Tracking', frame)

        # Press 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Release the video capture object.

cap.release()

# Close all windows.
cv2.destroyAllWindows()
