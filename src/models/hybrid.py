import torch
import torch.nn as nn
from src.models.unet import DoubleConv
from src.models.transformer import TransformerBlock

class UNetTransformer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=256, emb_size=256, depth=2):
        super(UNetTransformer, self).__init__()

        # UNet encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, emb_size)

        # Transformer bottleneck
        self.flatten = nn.Flatten(2)
        self.transformer = nn.ModuleList([TransformerBlock(emb_size=emb_size, num_heads=4) for _ in range(depth)])

        # Decoder
        self.up1 = nn.ConvTranspose2d(emb_size, 128, 2, stride=2)
        self.dec1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Transformer bottleneck
        B, C, H, W = e3.shape
        x_flat = e3.flatten(2).transpose(1, 2)  # [B, N, C]
        for blk in self.transformer:
            x_flat = blk(x_flat)
        x_trans = x_flat.transpose(1, 2).reshape(B, C, H, W)

        d1 = self.dec1(torch.cat([self.up1(x_trans), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return self.final(d2)

if __name__ == "__main__":
    model = UNetTransformer()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("UNetTransformer output:", y.shape)
