import cv2
import pyautogui


def Mouse(cap):
    print("in Mouse")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Palm_Fist Gesture Tracking', frame)
