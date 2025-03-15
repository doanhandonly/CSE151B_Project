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
        self.archive_root = os.path.join('archive', "Train_Alphabet" if split in ["train", "val"] else "test")
        self.mask_root = os.path.join('masks', "Train_Alphabet" if split in ["train", "val"] else "test")
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
        
        relative_path = os.path.relpath(img_path, self.archive_root)
        mask_path = os.path.join(self.mask_root, relative_path.replace(".png", "_mask.png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self.apply_augmentations(image, mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def apply_augmentations(self, image, mask):        
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        crop_size = (256, 256)  
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        return image, mask
