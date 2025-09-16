import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

IMG_EXT = [".png", ".jpg", ".jpeg", ".bmp"]


class MultiTaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, class_to_idx=None, transform=None, img_size=256):
        """
        img_dir: thư mục ảnh gốc (train/val/test)
        mask_dir: thư mục mask tương ứng (nếu có)
        class_to_idx: dict ánh xạ tên folder -> nhãn số
        transform: augmentation (albumentations / torchvision)
        img_size: resize ảnh về kích thước này
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size

        self.class_to_idx = class_to_idx if class_to_idx else {}
        self.samples = []

        # duyệt đệ quy qua tất cả file ảnh
        for root, _, files in os.walk(img_dir):
            class_name = os.path.basename(root)

            # bỏ qua folder rỗng hoặc không chứa ảnh
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

        img = cv2.resize(img, (self.img_size, self.img_size))

        # load mask nếu có
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
        else:
            mask = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)

        # chuyển sang tensor
        img = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        # augmentation nếu có
        if self.transform:
            # nếu bạn dùng albumentations thì nên chuyển sang apply ở đây
            pass

        return img, mask, label


def get_loader(img_dir, mask_dir=None, class_to_idx=None, batch_size=4, shuffle=True, img_size=256):
    dataset = MultiTaskDataset(img_dir, mask_dir, class_to_idx, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
