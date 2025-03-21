import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import multiprocessing
from tqdm import tqdm
import logging
import os


# Ensure the logs directory exists
log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)  # Creates logs/ if it doesn't exist

# Set up logging
log_file = os.path.join(log_dir, "corrupted_videos.log")
logging.basicConfig(filename=log_file, level=logging.ERROR)

# Enable GPU acceleration if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(f"GPU Error: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Load WLASL metadata
with open("../start_kit/WLASL_v0.3.json", "r") as f:
    wlasl_data = json.load(f)

# Create directory for keypoints
os.makedirs("../data/keypoints", exist_ok=True)

# Function to extract keypoints from video with frame skipping
logging.basicConfig(filename="../logs/corrupted_videos.log", level=logging.ERROR)

def extract_keypoints_from_video(video_path, output_path, frame_skip=2):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return

        keypoints_sequence = []
        frame_count = 0  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:  # Process every 2nd frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                keypoints = np.zeros((21, 2))  
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i, landmark in enumerate(hand_landmarks.landmark):
                            keypoints[i] = [landmark.x, landmark.y]

                keypoints_sequence.append(keypoints.flatten())  # (42,)

            frame_count += 1  # Increment counter

        cap.release()
        np.save(output_path, np.array(keypoints_sequence))  # Save as .npy

    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")

# Function to process a single video
def process_video(entry):
    gloss = entry["gloss"]
    for instance in entry["instances"]:
        video_id = instance["video_id"]
        video_path = f"../start_kit/raw_videos/{video_id}.mp4"
        output_path = f"../data/keypoints/{video_id}.npy"

        if os.path.exists(video_path) and not os.path.exists(output_path):
            extract_keypoints_from_video(video_path, output_path, frame_skip=2)  # Optimized

# Run multiprocessing
if __name__ == "__main__":
    with multiprocessing.Pool(processes=8) as pool:  # Adjust based on CPU cores
        list(tqdm(pool.imap(process_video, wlasl_data), total=len(wlasl_data), desc="Processing WLASL Videos"))