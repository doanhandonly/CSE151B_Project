import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import swa_utils

from config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config('configs.yaml')

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_classes=27):
        super().__init__()
        
        # self.encoder1 = self.conv_block(in_channels, 64)

        # self.encoder2 = self.conv_block(64, 128)
        # self.encoder3 = self.conv_block(128, 256)
        
        # self.bottleneck = self.conv_block(256, 512) 

        # self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.decoder3 = self.conv_block(512, 256)
        # self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.decoder2 = self.conv_block(256, 128)
        # self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.decoder1 = self.conv_block(128, 64)
        
        # self.final_conv = nn.Conv2d(512, out_channels, kernel_size=1)

        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)), 
        #     nn.Flatten(),
        #     nn.Linear(64, num_classes)
        # )
        # self.pool = nn.MaxPool2d(2)

        # self.encoder1 = self.conv_block(in_channels, 32)

        # self.encoder2 = self.conv_block(32, 64)
        # self.encoder3 = self.conv_block(64, 128)
        # self.encoder4 = self.conv_block(128, 256)

        self.encoder1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, dilation=1)
        self.encoder2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1)
        self.encoder3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, dilation=1)
        self.encoder4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.bnd1 = nn.BatchNorm2d(32)
        self.bnd2 = nn.BatchNorm2d(64)
        self.bnd3 = nn.BatchNorm2d(128)
        self.bnd4 = nn.BatchNorm2d(256)


        
        # self.final_conv = nn.Conv2d(256, out_channels, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        self.pool = nn.MaxPool2d(2)
    
    # def conv_block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(out_channels),
    #         nn.ReLU(inplace=True),
    #     )

    def forward(self, x):
        e1 = self.bnd1(self.relu(self.encoder1(x)))
        # p1 = self.pool(e1)
        e2 = self.bnd2(self.relu(self.encoder2(e1)))
        # p2 = self.pool(e2)
        e3 = self.bnd3(self.relu(self.encoder3(e2)))
        # p3 = self.pool(e3)
        e4 = self.bnd4(self.relu(self.encoder4(e3)))

        # features = e4
        # print(f"Feature stats: min={features.min().item()}, max={features.max().item()}, mean={features.mean().item()}")
        # class_output = self.classifier(e4)
        # print(f"Output stats: min={class_output.min().item()}, max={class_output.max().item()}, mean={class_output.mean().item()}")

        class_output = self.classifier(e4)
        return class_output
        # return output, class_output
    
 

class CustomModel(UNet):
    def __init__(self, in_channels = 3, out_channels = 1, num_classes = 27):
        super().__init__(in_channels, out_channels, num_classes)

        # self.swa_model = swa_utils.AveragedModel(self).to(device)
        # self.swa_active = False 
        # self.swa_start = int(config['epochs'] * .75)
        # self.swa_lr = 1e-4 

    def update_swa(self):
        self.swa_model.update_parameters(self)

    def swap_swa_weights(self):
        for param, swa_param in zip(self.parameters(), self.swa_model.parameters()):
            param.data = swa_param.data.clone()

    def get_swa_scheduler(self, optimizer):
        return swa_utils.SWALR(optimizer, swa_lr = self.swa_lr, anneal_epochs = 5)

    def update_bn(self, train_loader, device):
        swa_utils.update_bn(train_loader, self.swa_model, device = device)
