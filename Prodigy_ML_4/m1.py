import cv2

from model.Mouse import Mouse

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture a frame.
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Palm_Fist Gesture Tracking', frame)
    Mouse(cap)
