import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet, CustomModel
from dataloader import SegmentationDataset
from util import compute_mean_std
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup

ARCHIVE_ROOT = "archive"
MASK_ROOT = "masks"
BATCH_SIZE = 8
LR = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    print("We are using the GPU.")
else:
    print("We are using the CPU.")

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
val_dataset = SegmentationDataset("val", transform=image_transform, target_transform=mask_transform, augment=True)
test_dataset = SegmentationDataset("test", transform=image_transform, target_transform=mask_transform, augment=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = UNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
class_criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

def train():
    print('training baseline model')
    best_val_loss = float('inf') 

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, masks, labels in loop:
            images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)

            seg_output, class_output = model(images)  

            loss_seg = criterion(seg_output, masks)  
            loss_class = class_criterion(class_output, labels)  

            loss = loss_seg + loss_class  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")
        val_loss = validate()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

def custom_train():
    print('training custom model')
    model = CustomModel().to(DEVICE)
    best_val_loss = float('inf') 
    
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    swa_scheduler = model.get_swa_scheduler(optimizer)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, masks, labels in loop:
            images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)

            seg_output, class_output = model(images)  

            loss_seg = criterion(seg_output, masks)  
            loss_class = class_criterion(class_output, labels)  

            loss = loss_seg + loss_class  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch >= model.swa_start:
            if not model.swa_active:
                model.swa_active = True  
            model.update_swa()  
            swa_scheduler.step() 
        else:
            scheduler.step()  

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")
        val_loss = validate()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    model.swap_swa_weights()
    model.update_bn(train_loader, DEVICE)

def validate():
    model.eval()
    total_loss = 0
    loop = tqdm(val_loader, leave=True, desc="Validating")

    with torch.no_grad():
        for images, masks, labels in loop:
            images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)

            seg_output, class_output = model(images)
            loss_seg = criterion(seg_output, masks)
            loss_class = class_criterion(class_output, labels)
            loss = loss_class + loss_seg
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(val_loader):.4f}")
    return total_loss / len(val_loader)

def test():
    model.eval()

    model.load_state_dict(torch.load('best_model.pth'))
    print("Best model loaded for testing.")
    with torch.no_grad():
        for i, (image, mask, label) in enumerate(test_loader):
            print(i)
            image, mask, label = image.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            output, class_output = model(image)  

            predicted_class = torch.argmax(class_output, dim=1).item()
            predicted_letter = "Blank" if predicted_class == 26 else chr(predicted_class + ord('A'))

            torchvision.utils.save_image(image, f"results/input_{i}.png")
            torchvision.utils.save_image(mask, f"results/mask_{i}.png")
            torchvision.utils.save_image(output, f"results/output_{i}.png")

            print(f"Image {i}: Predicted Letter - {predicted_letter}")

if __name__ == "__main__":
    custom_train()
    test()