from torchvision import transforms
import glob
import numpy as np
import os
import torch
from PIL import Image
def compute_mean_std(dataset_path, split="train"):
    mean = np.zeros(3)
    std = np.zeros(3)
    num_pixels = 0
    
    for split in ['Train_Alphabet', 'Test_Alphabet']:
        split_path = os.path.join(dataset_path, split)
        
        for root, dirs, files in os.walk(split_path):
            for file in files:
                img_path = os.path.join(root, file)
                image = Image.open(img_path).convert("RGB")
                image = transforms.ToTensor()(image)
                
                mean += image.mean(dim=(1, 2)).numpy()
                std += image.std(dim=(1, 2)).numpy()
                num_pixels += 1

    mean /= num_pixels
    std /= num_pixels
    return mean, std

def class_accuracy(outputs, targets):
    arr = torch.argmax(outputs, dim=1)

    toret = arr == targets
    # print(arr, targets)
    return torch.sum(toret).item() / toret.nelement()