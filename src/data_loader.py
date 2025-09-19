import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_EXT = [".png", ".jpg", ".jpeg", ".bmp"]


# ---------------- Augmentation ----------------
def get_transforms(img_size=256, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),   # chuẩn hóa grayscale
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])


# ---------------- Dataset ----------------
class MultiTaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, class_to_idx=None,
                 transform=None, img_size=256):
        """
        img_dir: thư mục ảnh gốc (train/val/test)
        mask_dir: thư mục mask tương ứng (nếu có)
        class_to_idx: dict ánh xạ tên folder -> nhãn số
        transform: augmentation (albumentations)
        img_size: resize ảnh/mask về kích thước này
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size

        self.class_to_idx = class_to_idx if class_to_idx else {}
        self.samples = []

        # duyệt file ảnh
        for root, _, files in os.walk(img_dir):
            class_name = os.path.basename(root)
            valid_files = [f for f in files if os.path.splitext(f)[-1].lower() in IMG_EXT]
            if not valid_files:
                continue

            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)

            for fname in valid_files:
                img_path = os.path.join(root, fname)
                mask_path = None
                if mask_dir:
                    mask_path = os.path.join(mask_dir, class_name, fname)
                label = self.class_to_idx[class_name]
                self.samples.append((img_path, mask_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        # load ảnh
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"[ERROR] Không đọc được ảnh: {img_path}")

        # load mask
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = torch.zeros((self.img_size, self.img_size), dtype=torch.float32).numpy()

        # apply transform
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        else:
            # fallback: resize + tensor hóa
            img = cv2.resize(img, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            img = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)
        return img, mask, label


# ---------------- DataLoader ----------------
def get_loader(img_dir, mask_dir=None, class_to_idx=None,
               batch_size=4, shuffle=True, img_size=256, transform=None):
    dataset = MultiTaskDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        class_to_idx=class_to_idx,
        transform=transform,
        img_size=img_size
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
