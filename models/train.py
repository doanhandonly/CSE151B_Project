import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load preprocessed data
X = np.load("./data/X.npy")  # shape: (num_samples, max_seq_length, 84)
y = np.load("./data/y.npy")  # shape: (num_samples, num_classes)

with open("./data/label_mapping.json", "r") as f:
    label_mapping = json.load(f)

num_classes = len(label_mapping)
max_seq_length = X.shape[1]  # already padded
feature_dim = X.shape[2]     # should be 84 if using 2 hands

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of classes:", num_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define Bi-LSTM model
model = Sequential([
    Input(shape=(max_seq_length, feature_dim)),

    Bidirectional(LSTM(256, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    # Optional attention mechanism or a Dense "context" layer
    Dense(128, activation="tanh"),
    Dropout(0.3),

    LSTM(64),
    Dense(256, activation='relu'),
    Dropout(0.3),

    # IMPORTANT: match the actual number of distinct classes
    Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=1e-4)
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
model.fit(X_train, y_train, 
          validation_data=(X_test, y_test),
          epochs=100, batch_size=32,
          callbacks=[early_stopping])

# Save model
model.save("../models/asl_bilstm.keras")
print("Model training complete.")