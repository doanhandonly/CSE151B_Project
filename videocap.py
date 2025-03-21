from ultralytics import YOLO  # YOLO v8 model loading and inference
import cv2
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import swa_utils
from torch.utils.data import DataLoader
from model import UNet, CustomModel
from dataloader import SegmentationDataset
from util import compute_mean_std
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup

from util import class_accuracy

from PIL import Image

def capture_frames(video_path, output_folder, frame_interval=10):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0
    model_path = "yolov8n.pt"
    model = YOLO(model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        # Save frame every `frame_interval` frames
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            
            # print(f"Saved: {frame_filename}")

            input_image_path = frame_filename
            output_yolo_folder = "yolo"
            output_image_path = os.path.join(output_yolo_folder, f"frame_{saved_count:04d}.png")

            results = model.predict(input_image_path, conf=0.3)

            # Load image with OpenCV
            img = cv2.imread(input_image_path)

            # Draw bounding boxes
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    conf = box.conf[0].item()  # Confidence score
                    
                    # Draw rectangle & label
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                    # cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cropped_img = img[250:750, 750:1250]
                    cv2.imwrite(output_image_path, cropped_img)
                    # print(f"Box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {conf:.2f}")

            # Save the modified image with bounding boxes
            # cv2.imwrite(output_image_path, img)
            # print(f"Output image saved to {output_image_path}")

            saved_count += 1

        frame_count += 1        

    cap.release()
    print("Frame capture complete.")

def test():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Adjust based on model input size
        transforms.ToTensor(),
    ])

    model.load_state_dict(torch.load('best_model.pth'))
    print("Best model loaded for testing.")
    image_folder = "yolo"
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]
    # print(image_paths)
    with torch.no_grad():
        counter = 0
        for i, (image_path) in enumerate(image_paths):
            # print(i)
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(DEVICE)
            class_output = model(image)  

            predicted_class = torch.argmax(class_output, dim=1).item()
            predicted_letter = "Blank" if predicted_class == 26 else chr(predicted_class + ord('A'))

            # if label == predicted_class:
            #     counter += 1

            torchvision.utils.save_image(image, f"yoloout/input_{i}.png")
            # torchvision.utils.save_image(mask, f"results/mask_{i}.png")
            # torchvision.utils.save_image(output, f"results/output_{i}.png")

            print(f"Image {i}: Predicted Letter - {predicted_letter}")
        # print(f"Test Accuracy = {counter / len(test_loader):.4f}")

# Example usage
video_path = "ASLalphabet5.mp4"  # Change to your video path
output_folder = "frames"
capture_frames(video_path, output_folder)
test()