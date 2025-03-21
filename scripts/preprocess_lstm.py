import os
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load WLASL metadata
with open("../start_kit/WLASL_v0.3.json", "r") as f:
    wlasl_data = json.load(f)

# Collect dataset
X, y = [], []
labels = []

# Load keypoints
for entry in wlasl_data:
    gloss = entry["gloss"]
    labels.append(gloss)

    for instance in entry["instances"]:
        video_id = instance["video_id"]
        keypoints_path = f"../data/keypoints/{video_id}.npy"

        if os.path.exists(keypoints_path):
            keypoints = np.load(keypoints_path)
            X.append(keypoints)  # List of variable-length sequences
            y.append(gloss)

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded, num_classes=len(labels))

# 🔥 Fix: Pad all sequences to the same max length
max_seq_length = max(len(seq) for seq in X)  # Find the longest sequence
X_padded = pad_sequences(X, maxlen=max_seq_length, dtype="float32", padding="post", truncating="post")

# Save processed data
np.save("../data/X.npy", X_padded)  # Now it's a uniform shape
np.save("../data/y.npy", y_onehot)

# Save label mappings
with open("../data/label_mapping.json", "w") as f:
    json.dump({i: label for i, label in enumerate(labels)}, f)

print(f"Data saved: {len(X_padded)} samples, {len(labels)} unique signs. Sequence length: {max_seq_length}")