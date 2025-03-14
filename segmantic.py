import torchvision.models.segmentation as segmentation
import torch
import torchvision.transforms as transforms
import os
from PIL import Image
output_dir = 'masks'
root_dir = 'archive'
model = segmentation.deeplabv3_resnet101(pretrained='True')
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0, std=1)
])

for dataset_type in ["Test_Alphabet", "Train_Alphabet"]:
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        os.makedirs(os.path.join(output_dir, dataset_type, letter), exist_ok=True)

for dataset_type in ["Test_Alphabet", "Train_Alphabet"]:
    dataset_dir = os.path.join(root_dir, dataset_type)
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        letter_dir = os.path.join(dataset_dir, letter)
        for filename in os.listdir(letter_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(letter_dir, filename)

                input_image = Image.open(filepath).convert("RGB")
                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0)

                with torch.no_grad():
                    output = model(input_batch)['out'][0]

                predicted_mask = torch.argmax(output, dim=0)
                output_image = transforms.ToPILImage()(predicted_mask.byte())

                output_filename = os.path.splitext(filename)[0] + "_mask.png"
                output_filepath = os.path.join(output_dir, dataset_type, letter, output_filename)
                output_image.save(output_filepath)

print("Segmentation masks created and saved in:", output_dir) 