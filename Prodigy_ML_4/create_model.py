import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Load captured gesture data
gesture_data = np.load('gesture_data.npy')
gesture_labels = np.load('gesture_labels.npy')

# Normalize the data
gesture_data = gesture_data.reshape(gesture_data.shape[0], -1)

# Encode labels
label_encoder = LabelEncoder()
gesture_labels_encoded = label_encoder.fit_transform(gesture_labels)

# Convert labels to categorical
gesture_labels_categorical = to_categorical(gesture_labels_encoded)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(gesture_data, gesture_labels_categorical, test_size=0.2,
                                                    random_state=42)
# Build the neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(gesture_data.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('gesture_recognition_model.keras')
print(np.unique(gesture_labels))
print([np.sum(gesture_labels == x) for x in np.unique(gesture_labels)])
