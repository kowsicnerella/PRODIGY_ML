import numpy as np
from tensorflow.keras.models import load_model
from create_model import label_encoder, gesture_data, X_test, y_test


# Function to predict gesture
def predict_gesture(model_predict, landmarks):
    print(landmarks.shape)
    landmarks = np.array(landmarks).reshape(1, -1)
    # landmarks = landmarks / np.max(landmarks)
    prediction = model_predict.predict(landmarks)
    gesture = label_encoder.inverse_transform([np.argmax(prediction)])
    return gesture[0]


if __name__ == "__main__":
    # Example to predict a gesture
    model = load_model('gesture_recognition_model.keras')

    landmark = gesture_data[1001]  # Replace with actual landmark data
    predicted_gesture = predict_gesture(model, landmark)
    print(f"Predicted gesture: {predicted_gesture}")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
