import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Function to load images and labels from a folder
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.resize(img, (64, 64))  # Resize images to a fixed size
            images.append(img)
            labels.append(label)
    print(f"{folder} image loading done!")
    return images, labels


# Load cat and dog images
print("loading cat images...")
cat_images, cat_labels = load_images_from_folder('dogs-vs-cats/train/cat', 0)
print("loading dog images...")
dog_images, dog_labels = load_images_from_folder('dogs-vs-cats/train/dog', 1)

# Combine images and labels
X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

# Flatten the images
n_samples, h, w, c = X.shape
X_flattened = X.reshape(n_samples, -1)  # Flatten each image

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

print("Splitting test-set done!")
# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
print("Training SVM model...")
clf = SVC(kernel='linear')  # You can experiment with other kernels like 'rbf'
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
