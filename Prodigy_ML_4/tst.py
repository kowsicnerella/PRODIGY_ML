import math
import time

import cv2
import mouse
import numpy as np
import pyautogui
# from tensorflow.keras.models import load_model

from model.gesture_control import get_active_window_title

# from model.model_predict import predict_gesture

# model = load_model("gesture_recognition_model.keras")
# res = predict_gesture(gesture_data[1000])
# print("res: ", res)

gesture_data = np.load('gesture_data.npy')
gesture_labels = np.load('gesture_labels.npy')
print(np.unique(gesture_labels))
print([np.sum(gesture_labels == x) for x in np.unique(gesture_labels)])

print(gesture_labels[2999])
print(gesture_labels[2000])

# gesture_labels = np.delete(gesture_labels, range(2000, 2999), axis=0)
# gesture_data = np.delete(gesture_data, range(2000, 2999), axis=0)

print(np.unique(gesture_labels))
print([np.sum(gesture_labels == x) for x in np.unique(gesture_labels)])

'''np.save('gesture_labels.npy', gesture_labels)
np.save('gesture_data.npy', gesture_data)'''
pyautogui.FAILSAFE = False

'''def calculate_duration(prev_x, prev_y, curr_x, curr_y):
    """Calculate duration based on the distance between two points."""
    min_duration, max_duration = 0.5, 0.01
    distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
    # Example scaling: max_duration for distance 0, min_duration for a large distance (e.g., 1000 pixels)
    max_distance = 1000  # Adjust this value based on your requirements
    duration = max_duration - (max_duration - min_duration) * min(distance / max_distance, 1)
    return duration


duration = calculate_duration(0.123, 0.134, 527, 913)

print(duration)
pyautogui.moveTo(0.123, 0.134)'''

'''
screen_width, screen_height = pyautogui.size()
# mouse.move(109.85494136810303 * screen_width * 2, 328.12838315963745 * screen_height * 2, False, duration=1)
mouse.move(199.3969030380249, 651.8006086349487, True, duration=1)
mouse.release()
mouse.move(10, 10, False, duration=1)

lst = set()
while 1:
    lst.add(get_active_window_title())
    print(lst)
'''