import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import random

class SegmentationDataset(Dataset):
    def __init__(self, split="train", transform=None, target_transform=None, augment=False):
        self.archive_root = os.path.join('archive', "Train_Alphabet" if split in ["train", "val"] else "Test_Alphabet")
        self.mask_root = os.path.join('masks', "Train_Alphabet" if split in ["train", "val"] else "Test_Alphabet")
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment 
        
        self.image_paths = sorted(glob.glob(os.path.join(self.archive_root, "*", "*.rgb_0000.png")))
        
        if split == "train":
            self.image_paths = self.image_paths[:int(0.8 * len(self.image_paths))]
        elif split == "val":
            self.image_paths = self.image_paths[int(0.8 * len(self.image_paths)):]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Extract the letter from the folder name (including "Blank")
        letter = os.path.basename(os.path.dirname(img_path))

        relative_path = os.path.relpath(img_path, self.archive_root)
        mask_path = os.path.join(self.mask_root, relative_path.replace(".png", "_mask.png"))

        image = Image.open(img_path).convert("RGB")

        # Handle "Blank" case: create an empty mask if it doesn't exist
        if letter == "Blank":
            mask = Image.new("L", image.size, 0)  # All zeros mask
        else:
            mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self.apply_augmentations(image, mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # Convert letter to a numerical class index, including "Blank"
        letter_to_index = {chr(i): i - ord('A') for i in range(ord('A'), ord('Z') + 1)}
        letter_to_index["Blank"] = 26  # Assign index 26 for "Blank" class

        letter_idx = letter_to_index.get(letter, -1)  # Assign -1 if error

        return image, mask, letter_idx

    def apply_augmentations(self, image, mask):        
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        crop_size = (256, 256)  
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        return image, mask