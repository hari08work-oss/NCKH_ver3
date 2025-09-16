import torch
import torch.nn as nn
from src.models.unet import UNet

class MultiTaskNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiTaskNet, self).__init__()
        self.unet = UNet(in_channels=1, out_channels=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1, num_classes)  # giả sử lấy feature map cuối

    def forward(self, x):
        mask = self.unet(x)   # segmentation
        feat = self.pool(mask)   # dùng mask feature làm input cho classification
        feat = feat.view(feat.size(0), -1)
        cls = self.fc(feat)
        return mask, cls
