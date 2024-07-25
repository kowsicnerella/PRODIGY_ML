import numpy as np

# Load data and labels
gesture_data = np.load('gesture_data.npy')
gesture_labels = np.load('gesture_labels.npy')

# Create a boolean mask where the label is 'up'
mask = gesture_labels != 'up'

# Apply the mask to filter out the data and labels where the label is 'up'
filtered_gesture_data = gesture_data[mask]
filtered_gesture_labels = gesture_labels[mask]

# Save the filtered data and labels if needed
np.save('filtered_gesture_data.npy', filtered_gesture_data)
np.save('filtered_gesture_labels.npy', filtered_gesture_labels)

# Optional: Print the shapes to verify
print(f'Original data shape: {gesture_data.shape}')
print(f'Filtered data shape: {filtered_gesture_data.shape}')
print(f'Original labels shape: {gesture_labels.shape}')
print(f'Filtered labels shape: {filtered_gesture_labels.shape}')

print(np.unique(gesture_labels))
print([np.sum(gesture_labels == x) for x in np.unique(gesture_labels)])

print(np.unique(filtered_gesture_labels))
print([np.sum(filtered_gesture_labels == x) for x in np.unique(filtered_gesture_labels)])

np.save('gesture_data.npy', filtered_gesture_data)
np.save('gesture_labels.npy', filtered_gesture_labels)
