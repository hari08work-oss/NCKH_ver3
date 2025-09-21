import os
import matplotlib.pyplot as plt
from src.data_loader import get_loader

DATA_DIR = "data/dataset_split"

if __name__ == "__main__":
    # tạo dataloader
    train_loader = get_loader(
        os.path.join(DATA_DIR, "train/images"),
        os.path.join(DATA_DIR, "train/masks"),
        batch_size=2, img_size=256, shuffle=True
    )

    print(f"[INFO] Số batch train: {len(train_loader)}")

    # lấy 1 batch thử
    batch = next(iter(train_loader))

    # xử lý theo số phần tử mà dataset trả về
    if len(batch) == 2:
        imgs, masks = batch
        labels = None
    elif len(batch) == 3:
        imgs, masks, labels = batch
    else:
        raise RuntimeError(f"[ERROR] Dataloader trả về {len(batch)} giá trị, không đúng định dạng.")

    # in shape
    print(f"[INFO] Batch images shape: {imgs.shape}")
    print(f"[INFO] Batch masks shape : {masks.shape}")
    if labels is not None:
        print(f"[INFO] Batch labels shape: {labels.shape}")

    # hiển thị 1 ảnh và mask
    img = imgs[0].squeeze().numpy()
    mask = masks[0].squeeze().numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")

    plt.show()
