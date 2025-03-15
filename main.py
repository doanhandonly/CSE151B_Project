import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from dataloader import SegmentationDataset
from util import compute_mean_std
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm


ARCHIVE_ROOT = "archive"
MASK_ROOT = "masks"
BATCH_SIZE = 8
LR = 0.001
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#mean, std = compute_mean_std(ARCHIVE_ROOT, split="train")
#print(mean, std)
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.49145114, 0.46978662, 0.42543335], std=[0.15824795, 0.16471928, 0.17677158])
])

mask_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = SegmentationDataset("train", transform=image_transform, target_transform=mask_transform, augment=True)
val_dataset = SegmentationDataset("val", transform=image_transform, target_transform=mask_transform, augment=False)
test_dataset = SegmentationDataset("test", transform=image_transform, target_transform=mask_transform, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = UNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train():
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")
        validate()

def validate():
    model.eval()
    total_loss = 0
    loop = tqdm(val_loader, leave=True, desc="Validating")

    with torch.no_grad():
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(val_loader):.4f}")

def test():
    model.eval()
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            output = torch.sigmoid(model(image))  
            
            torchvision.utils.save_image(image, f"results/input_{i}.png")
            torchvision.utils.save_image(mask, f"results/mask_{i}.png")
            torchvision.utils.save_image(output, f"results/output_{i}.png")

if __name__ == "__main__":
    train()
    test()
