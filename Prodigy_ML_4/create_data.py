import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create a video capture object
cap = cv2.VideoCapture(0)


# Function to capture data
def capture_gesture_data(num_samples, gesture_name):
    gesture_data = []
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        sample_count = 0
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    h, w, c = frame.shape
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y])
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    gesture_data.append(landmarks)
                    sample_count += 1

            cv2.imshow('Capture Gesture Data', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    gesture_data = np.array(gesture_data)
    labels = np.array([gesture_name] * num_samples)
    return gesture_data, labels


# Example to capture data for a specific gesture
gesture_data, gesture_labels = capture_gesture_data(1000, 'up')

'''
# Save data to a file or combine with other gestures
np.save('gesture_data.npy', gesture_data)
np.save('gesture_labels.npy', gesture_labels)
'''

# Load existing labels (assuming they are already in an array)
existing_labels = np.load('gesture_labels.npy')
existing_data = np.load('gesture_data.npy')

# Combine existing and new label
combined_data = np.concatenate((existing_data, gesture_data), axis=0)
combined_labels = np.concatenate((existing_labels, gesture_labels), axis=0)

# Save the combined labels
np.save('gesture_data.npy', combined_data)
np.save('gesture_labels.npy', combined_labels)

print("Successfully appended new labels to the file.")
