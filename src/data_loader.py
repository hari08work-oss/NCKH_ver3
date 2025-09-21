import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path

IMG_EXT = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

def get_transforms(img_size=256, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])

class MultiTaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, class_to_idx=None, transform=None, img_size=256):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.transform = transform
        self.img_size  = img_size
        self.class_to_idx = class_to_idx if class_to_idx else {}
        self.samples = []

        if not self.img_dir.exists():
            raise RuntimeError(f"[Dataset] img_dir không tồn tại: {self.img_dir}")

        # Duyệt theo class con nếu có, nếu không thì đọc flat
        subdirs = [d for d in self.img_dir.iterdir() if d.is_dir()]
        if subdirs:
            for sub in sorted(subdirs):
                class_name = sub.name
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                for fp in sub.rglob("*"):
                    if fp.suffix.lower() in IMG_EXT:
                        img_path = fp
                        mask_path = (self.mask_dir / class_name / fp.name) if self.mask_dir else None
                        self.samples.append((img_path, mask_path, self.class_to_idx[class_name]))
        else:
            # flat
            class_name = "default"
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = 0
            for fp in self.img_dir.rglob("*"):
                if fp.suffix.lower() in IMG_EXT:
                    img_path = fp
                    mask_path = (self.mask_dir / fp.name) if self.mask_dir else None
                    self.samples.append((img_path, mask_path, self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"[Dataset EMPTY] Không thấy ảnh hợp lệ trong: {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"[ERROR] Không đọc được ảnh: {img_path}")

        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros_like(img, dtype=np.uint8)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
            if mask.ndim == 2:  # HxW -> 1xHxW
                mask = mask.unsqueeze(0)
            mask = (mask > 127).float()
        else:
            img  = cv2.resize(img, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            img  = torch.tensor(img/255.0, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor((mask>127).astype(np.float32)).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)
        return img, mask, label

def get_loader(img_dir, mask_dir=None, class_to_idx=None, batch_size=4, shuffle=True, img_size=256, transform=None):
    dataset = MultiTaskDataset(
        img_dir=img_dir, mask_dir=mask_dir, class_to_idx=class_to_idx,
        transform=transform if transform else get_transforms(img_size, is_train=shuffle),
        img_size=img_size
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)