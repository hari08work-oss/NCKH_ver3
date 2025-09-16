import torch
import torch.nn as nn


# --- Patch Embedding ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_size=256, img_size=256):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, emb_size, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_emb = nn.Parameter(
            torch.randn((img_size // patch_size) ** 2 + 1, emb_size)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)                       # [B, emb, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)       # [B, N, emb]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, emb]
        x = x + self.pos_emb
        return x


# --- Transformer Encoder Block ---
class TransformerBlock(nn.Module):
    def __init__(self, emb_size=256, num_heads=4, dropout=0.1, forward_expansion=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_size, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _x = self.norm1(x)
        attn_out, _ = self.attn(_x, _x, _x)
        x = x + self.dropout(attn_out)
        _x = self.norm2(x)
        ff_out = self.ff(_x)
        x = x + self.dropout(ff_out)
        return x


# --- TransUNet Simplified ---
class TransUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=256, patch_size=16, emb_size=256, depth=4):
        super(TransUNet, self).__init__()
        self.embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.encoder = nn.ModuleList(
            [TransformerBlock(emb_size=emb_size, num_heads=4) for _ in range(depth)]
        )

        # Decoder: 16x16 -> 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(emb_size, 256, 2, stride=2),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),       # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),        # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),         # 128 -> 256
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 1)                   # Final mask
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_size = emb_size

    def forward(self, x):
        B = x.shape[0]
        x = self.embed(x)  # [B, N, emb]
        for blk in self.encoder:
            x = blk(x)

        # Bỏ cls token, reshape thành feature map
        x = x[:, 1:, :]  # remove cls
        h = w = self.img_size // self.patch_size
        x = x.transpose(1, 2).reshape(B, self.emb_size, h, w)
        return self.decoder(x)


if __name__ == "__main__":
    model = TransUNet()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
