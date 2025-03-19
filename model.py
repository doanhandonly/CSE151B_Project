import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import swa_utils

from config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config('configs.yaml')

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_classes=27):
        super(UNet, self).__init__()
        
        self.encoder1 = self.conv_block(in_channels, 64)

        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        
        self.bottleneck = self.conv_block(256, 512) 

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        b = self.bottleneck(p3)

        d3 = self.upconv3(b)
        d3 = crop_tensor(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = crop_tensor(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = crop_tensor(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        class_output = self.classifier(d1)
        output = self.final_conv(d1)
        return output, class_output
    
def crop_tensor(target, reference):
    _, _, h1, w1 = target.shape
    _, _, h2, w2 = reference.shape
    diff_h = h1 - h2
    diff_w = w1 - w2
    target = target[:, :, :h1 - diff_h, :w1 - diff_w]
    return target

class CustomModel(UNet):
    def __init__(self, in_channels = 3, out_channels = 1, num_classes = 27):
        super().__init__(in_channels, out_channels, num_classes)

        self.swa_model = swa_utils.AveragedModel(self).to(device)
        self.swa_active = False 
        self.swa_start = int(config['epochs'] * .75)
        self.swa_lr = 1e-4 

    def update_swa(self):
        self.swa_model.update_parameters(self)

    def swap_swa_weights(self):
        for param, swa_param in zip(self.parameters(), self.swa_model.parameters()):
            param.data = swa_param.data.clone()

    def get_swa_scheduler(self, optimizer):
        return swa_utils.SWALR(optimizer, swa_lr = self.swa_lr, anneal_epochs = 5)

    def update_bn(self, train_loader, device):
        swa_utils.update_bn(train_loader, self.swa_model, device = device)