# import argparse, json, time, os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm

# from src.config import *
# from src.data_loader import get_loader
# from src.utils import dice_score, iou_score, save_checkpoint


# # ---------------- Loss Functions ----------------
# def dice_loss(pred, target, smooth=1.0):
#     """Dice Loss cơ bản (0 ≤ loss ≤ 1)"""
#     pred = torch.sigmoid(pred)   # output model chưa qua sigmoid
#     intersection = (pred * target).sum()
#     dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
#     return 1 - dice   # Dice Loss = 1 - Dice coefficient


# class BCEDiceLoss(nn.Module):
#     """Kết hợp BCE + Dice Loss"""
#     def __init__(self, bce_weight=0.5):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.bce_weight = bce_weight

#     def forward(self, pred, target):
#         bce = self.bce(pred, target)       # BCE ≥ 0
#         dsc = dice_loss(pred, target)      # Dice Loss ≥ 0
#         return self.bce_weight * bce + (1 - self.bce_weight) * dsc


# # ---------------- Training ----------------
# def train(epochs, model_class, use_best_config=True, best_config_path="outputs/best_config.json"):
#     # ----- load config -----
#     cfg = None
#     if model_class.__name__ == "TransUNet" and use_best_config and os.path.exists(best_config_path):
#         with open(best_config_path, "r", encoding="utf-8") as f:
#             obj = json.load(f)
#             cfg = obj["config"]
#             print("[TRAIN] Loaded best_config:", cfg)
#     else:
#         cfg = {
#             "lr": LR,
#             "weight_decay": 1e-4,
#             "depth": 4,
#             "num_heads": 4,
#             "emb_size": 256,
#             "patch_size": 16,
#             "img_size": 256,      # 👉 mặc định dùng 256 để đỡ tốn GPU
#             "batch_size": 1       # 👉 batch_size=1 cho GPU 4GB
#         }
#         print("[TRAIN] Using default config:", cfg)

#     # ----- model + optimizer + loss -----
#     if model_class.__name__ == "UNet":
#         model = model_class(in_channels=1, out_channels=1).to(DEVICE)
#     else:  # Transformer
#         model = model_class(
#             in_channels=1,
#             out_channels=1,
#             img_size=cfg["img_size"],
#             patch_size=cfg["patch_size"],
#             emb_size=cfg["emb_size"],
#             depth=cfg["depth"],
#         ).to(DEVICE)

#     optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

#     # 👉 dùng BCE + Dice
#     criterion = BCEDiceLoss(bce_weight=0.7).to(DEVICE)

#     # ----- data -----
#     train_loader = get_loader(
#         os.path.join(DATA_DIR, "train/images"),
#         os.path.join(DATA_DIR, "train/masks"),
#         batch_size=cfg["batch_size"], img_size=cfg["img_size"]
#     )

#     val_loader = get_loader(
#         os.path.join(DATA_DIR, "val/images"),
#         os.path.join(DATA_DIR, "val/masks"),
#         batch_size=cfg["batch_size"], shuffle=False, img_size=cfg["img_size"]
#     )

#     # Nếu muốn test loader:
#     # test_loader = get_loader(
#     #     os.path.join(DATA_DIR, "test/images"),
#     #     os.path.join(DATA_DIR, "test/masks"),
#     #     batch_size=1, shuffle=False, img_size=cfg["img_size"]
#     # )

#     best_dice = 0.0
#     total_start = time.time()

#     for epoch in range(1, epochs + 1):
#         epoch_start = time.time()
#         model.train()
#         total_loss, total_dice, total_iou = 0, 0, 0

#         loop = tqdm(train_loader, leave=False)
#         loop.set_description(f"Epoch [{epoch}/{epochs}]")

#         for imgs, masks, _ in loop:
#             imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

#             preds = model(imgs)
#             if preds.shape[2:] != masks.shape[2:]:
#                 preds = torch.nn.functional.interpolate(
#                     preds, size=masks.shape[2:], mode="bilinear", align_corners=False
#                 )

#             # ----- tính loss -----
#             loss = criterion(preds, masks)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             total_dice += dice_score(preds, masks).item()   # metric Dice (0→1)
#             total_iou += iou_score(preds, masks).item()     # metric IoU (0→1)
#             loop.set_postfix(loss=max(loss.item(), 0.0))    # clamp để log không âm

#         avg_loss = total_loss / len(train_loader)
#         avg_dice = total_dice / len(train_loader)
#         avg_iou = total_iou / len(train_loader)

#         # ----- validation -----
#         model.eval()
#         val_loss, val_dice, val_iou = 0, 0, 0
#         with torch.no_grad():
#             for imgs, masks, _ in val_loader:
#                 imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
#                 preds = model(imgs)
#                 if preds.shape[2:] != masks.shape[2:]:
#                     preds = torch.nn.functional.interpolate(
#                         preds, size=masks.shape[2:], mode="bilinear", align_corners=False
#                     )

#                 loss = criterion(preds, masks)
#                 val_loss += loss.item()
#                 val_dice += dice_score(preds, masks).item()
#                 val_iou += iou_score(preds, masks).item()

#         val_loss /= len(val_loader)
#         val_dice /= len(val_loader)
#         val_iou /= len(val_loader)
#         epoch_time = time.time() - epoch_start

#         print(f"\n[Epoch {epoch}/{epochs}] "
#               f"Train Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f} | "
#               f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f} | "
#               f"Time: {epoch_time:.2f}s")

#         if val_dice > best_dice:
#             best_dice = val_dice
#             save_checkpoint(model, optimizer, epoch, f"{CHECKPOINT_DIR}/best_model.pth")
#             print(f"[INFO] Saved BEST (Dice={best_dice:.4f})")

#     total_time = time.time() - total_start
#     print(f"\n[TRAIN DONE] Total: {total_time/60:.2f} min")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--epochs", type=int, default=EPOCHS, help="Số epoch huấn luyện")
#     parser.add_argument("--no_best_config", action="store_true", help="Không dùng best_config.json")
#     parser.add_argument("--model", type=str, default="transformer", choices=["unet", "transformer"], help="Chọn mô hình train")

#     args = parser.parse_args()

#     if args.model == "unet":
#         print("[INFO] Training baseline UNet...")
#         from src.models.unet import UNet
#         model_class = UNet
#     else:
#         print("[INFO] Training Transformer (TransUNet)...")
#         from src.models.transformer import TransUNet
#         model_class = TransUNet

#     train(args.epochs, model_class, use_best_config=not args.no_best_config)




import argparse, json, time, os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

from src.config import *
from src.data_loader import get_loader
from src.utils import dice_score, iou_score, save_checkpoint


# ---------------- Loss Functions ----------------
def dice_loss(pred, target, smooth=1.0):
    """Dice Loss cơ bản (0 ≤ loss ≤ 1)"""
    pred = torch.sigmoid(pred)   # output model chưa qua sigmoid
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


class BCEDiceLoss(nn.Module):
    """Kết hợp BCE + Dice Loss"""
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dsc = dice_loss(pred, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dsc


# ---------------- Multi-dataset Loader ----------------
def get_multi_loader(dataset_roots, split="train", batch_size=1, img_size=256, shuffle=True):
    """
    Gom nhiều dataset (mỗi dataset_root là thư mục chính chứa train/images, train/masks).
    Trả về 1 DataLoader duy nhất.
    """
    datasets = []
    for root in dataset_roots:
        img_dir = os.path.join(root, split, "images")
        mask_dir = os.path.join(root, split, "masks")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            raise RuntimeError(f"[Dataset ERROR] Không tìm thấy {img_dir} hoặc {mask_dir}")

        loader_tmp = get_loader(
            img_dir, mask_dir,
            batch_size=1, shuffle=False, img_size=img_size
        )
        datasets.append(loader_tmp.dataset)

    # gộp các dataset
    full_dataset = ConcatDataset(datasets)
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


# ---------------- Training ----------------
def train(epochs, model_class, dataset_roots, use_best_config=True, best_config_path="outputs/best_config.json"):
    # ----- load config -----
    if model_class.__name__ == "TransUNet" and use_best_config and os.path.exists(best_config_path):
        with open(best_config_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            cfg = obj["config"]
            print("[TRAIN] Loaded best_config:", cfg)
    else:
        cfg = {
            "lr": LR,
            "weight_decay": 1e-4,
            "depth": 4,
            "num_heads": 4,
            "emb_size": 256,
            "patch_size": 16,
            "img_size": 256,
            "batch_size": 1
        }
        print("[TRAIN] Using default config:", cfg)

    # ----- model + optimizer + loss -----
    if model_class.__name__ == "UNet":
        model = model_class(in_channels=1, out_channels=1).to(DEVICE)
    else:
        model = model_class(
            in_channels=1,
            out_channels=1,
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            emb_size=cfg["emb_size"],
            depth=cfg["depth"],
        ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = BCEDiceLoss(bce_weight=0.7).to(DEVICE)

    # ----- data (multi-dataset) -----
    train_loader = get_multi_loader(dataset_roots, "train",
                                    batch_size=cfg["batch_size"], img_size=cfg["img_size"], shuffle=True)
    val_loader = get_multi_loader(dataset_roots, "val",
                                  batch_size=cfg["batch_size"], img_size=cfg["img_size"], shuffle=False)

    best_dice = 0.0
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss, total_dice, total_iou = 0, 0, 0

        loop = tqdm(train_loader, leave=False)
        loop.set_description(f"Epoch [{epoch}/{epochs}]")

        for imgs, masks, _ in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            if preds.shape[2:] != masks.shape[2:]:
                preds = torch.nn.functional.interpolate(
                    preds, size=masks.shape[2:], mode="bilinear", align_corners=False
                )

            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score(preds, masks).item()
            total_iou += iou_score(preds, masks).item()
            loop.set_postfix(loss=max(loss.item(), 0.0))

        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        avg_iou = total_iou / len(train_loader)

        # ----- validation -----
        model.eval()
        val_loss, val_dice, val_iou = 0, 0, 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                if preds.shape[2:] != masks.shape[2:]:
                    preds = torch.nn.functional.interpolate(
                        preds, size=masks.shape[2:], mode="bilinear", align_corners=False
                    )

                loss = criterion(preds, masks)
                val_loss += loss.item()
                val_dice += dice_score(preds, masks).item()
                val_iou += iou_score(preds, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        epoch_time = time.time() - epoch_start

        print(f"\n[Epoch {epoch}/{epochs}] "
              f"Train Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f} | "
              f"Time: {epoch_time:.2f}s")

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, f"{CHECKPOINT_DIR}/best_model.pth")
            print(f"[INFO] Saved BEST (Dice={best_dice:.4f})")

    total_time = time.time() - total_start
    print(f"\n[TRAIN DONE] Total: {total_time/60:.2f} min")


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Số epoch huấn luyện")
    parser.add_argument("--no_best_config", action="store_true", help="Không dùng best_config.json")
    parser.add_argument("--model", type=str, default="transformer", choices=["unet", "transformer"], help="Chọn mô hình train")
    args = parser.parse_args()

    # Danh sách dataset gốc (không bị lặp path)
    dataset_roots = [
        os.path.join(DATA_DIR, "dataset_split"),
    ]

    if args.model == "unet":
        print("[INFO] Training baseline UNet...")
        from src.models.unet import UNet
        model_class = UNet
    else:
        print("[INFO] Training Transformer (TransUNet)...")
        from src.models.transformer import TransUNet
        model_class = TransUNet

    train(args.epochs, model_class, dataset_roots, use_best_config=not args.no_best_config)




# #ver_3 
# import argparse, json, time, os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from torch.utils.data import ConcatDataset, DataLoader

# from src.config import *
# from src.data_loader import get_loader
# from src.utils import dice_score, iou_score, save_checkpoint


# # ---------------- Loss Functions ----------------
# def dice_loss(pred, target, smooth=1.0):
#     pred = torch.sigmoid(pred)
#     intersection = (pred * target).sum()
#     dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
#     return 1 - dice


# class BCEDiceLoss(nn.Module):
#     def __init__(self, bce_weight=0.5):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.bce_weight = bce_weight

#     def forward(self, pred, target):
#         bce = self.bce(pred, target)
#         dsc = dice_loss(pred, target)
#         return self.bce_weight * bce + (1 - self.bce_weight) * dsc


# # ---------------- Multi-dataset Loader ----------------
# def get_multi_loader(dataset_roots, split="train", batch_size=1, img_size=256, shuffle=True):
#     datasets = []
#     for root in dataset_roots:
#         img_dir = os.path.join(root, f"{split}/images")
#         mask_dir = os.path.join(root, f"{split}/masks")

#         loader_tmp = get_loader(
#             img_dir, mask_dir,
#             batch_size=1,
#             shuffle=False, img_size=img_size
#         )
#         datasets.append(loader_tmp.dataset)

#     full_dataset = ConcatDataset(datasets)
#     loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)
#     return loader


# # ---------------- Training ----------------
# def train(epochs, model_class, dataset_roots, use_best_config=True, best_config_path="outputs/best_config.json"):
#     if use_best_config and os.path.exists(best_config_path):
#         with open(best_config_path, "r", encoding="utf-8") as f:
#             obj = json.load(f)
#             cfg = obj["config"]
#             print("[TRAIN] Loaded best_config:", cfg)
#     else:
#         cfg = {
#             "lr": LR,
#             "weight_decay": 1e-4,
#             "depth": 4,
#             "num_heads": 4,
#             "emb_size": 256,
#             "patch_size": 16,
#             "img_size": 256,
#             "batch_size": 1
#         }
#         print("[TRAIN] Using default config:", cfg)

#     # ----- model -----
#     model = model_class(
#         in_channels=1,
#         out_channels=1,
#         img_size=cfg["img_size"],
#         patch_size=cfg["patch_size"],
#         emb_size=cfg["emb_size"],
#         depth=cfg["depth"],
#     ).to(DEVICE)

#     optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

#     # Loss cho segmentation + classification
#     criterion_seg = BCEDiceLoss(bce_weight=0.7).to(DEVICE)
#     criterion_cls = nn.CrossEntropyLoss().to(DEVICE)

#     # ----- data -----
#     train_loader = get_multi_loader(dataset_roots, "train",
#                                     batch_size=cfg["batch_size"], img_size=cfg["img_size"], shuffle=True)
#     val_loader = get_multi_loader(dataset_roots, "val",
#                                   batch_size=cfg["batch_size"], img_size=cfg["img_size"], shuffle=False)

#     best_dice = 0.0
#     total_start = time.time()

#     for epoch in range(1, epochs + 1):
#         epoch_start = time.time()
#         model.train()
#         total_loss, total_dice, total_iou, total_acc = 0, 0, 0, 0

#         loop = tqdm(train_loader, leave=False)
#         loop.set_description(f"Epoch [{epoch}/{epochs}]")

#         for imgs, masks, labels in loop:
#             imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)

#             seg_out, cls_out = model(imgs)

#             # Segmentation loss
#             if seg_out.shape[2:] != masks.shape[2:]:
#                 seg_out = torch.nn.functional.interpolate(
#                     seg_out, size=masks.shape[2:], mode="bilinear", align_corners=False
#                 )
#             loss_seg = criterion_seg(seg_out, masks)

#             # Classification loss
#             loss_cls = criterion_cls(cls_out, labels)

#             loss = loss_seg + 0.3 * loss_cls

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             total_dice += dice_score(seg_out, masks).item()
#             total_iou += iou_score(seg_out, masks).item()

#             preds_cls = torch.argmax(cls_out, dim=1)
#             total_acc += (preds_cls == labels).float().mean().item()

#             loop.set_postfix(loss=max(loss.item(), 0.0))

#         avg_loss = total_loss / len(train_loader)
#         avg_dice = total_dice / len(train_loader)
#         avg_iou = total_iou / len(train_loader)
#         avg_acc = total_acc / len(train_loader)

#         # ----- validation -----
#         model.eval()
#         val_loss, val_dice, val_iou, val_acc = 0, 0, 0, 0
#         with torch.no_grad():
#             for imgs, masks, labels in val_loader:
#                 imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)

#                 seg_out, cls_out = model(imgs)
#                 if seg_out.shape[2:] != masks.shape[2:]:
#                     seg_out = torch.nn.functional.interpolate(
#                         seg_out, size=masks.shape[2:], mode="bilinear", align_corners=False
#                     )

#                 loss_seg = criterion_seg(seg_out, masks)
#                 loss_cls = criterion_cls(cls_out, labels)
#                 loss = loss_seg + 0.3 * loss_cls

#                 val_loss += loss.item()
#                 val_dice += dice_score(seg_out, masks).item()
#                 val_iou += iou_score(seg_out, masks).item()
#                 preds_cls = torch.argmax(cls_out, dim=1)
#                 val_acc += (preds_cls == labels).float().mean().item()

#         val_loss /= len(val_loader)
#         val_dice /= len(val_loader)
#         val_iou /= len(val_loader)
#         val_acc /= len(val_loader)
#         epoch_time = time.time() - epoch_start

#         print(f"\n[Epoch {epoch}/{epochs}] "
#               f"Train Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, Acc: {avg_acc:.4f} | "
#               f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, Acc: {val_acc:.4f} | "
#               f"Time: {epoch_time:.2f}s")

#         if val_dice > best_dice:
#             best_dice = val_dice
#             save_checkpoint(model, optimizer, epoch, f"{CHECKPOINT_DIR}/best_model.pth")
#             print(f"[INFO] Saved BEST (Dice={best_dice:.4f})")

#     total_time = time.time() - total_start
#     print(f"\n[TRAIN DONE] Total: {total_time/60:.2f} min")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--epochs", type=int, default=EPOCHS, help="Số epoch huấn luyện")
#     parser.add_argument("--no_best_config", action="store_true", help="Không dùng best_config.json")
#     args = parser.parse_args()

#     dataset_roots = [
#         os.path.join(DATA_DIR, "dataset_split"),
#         os.path.join(DATA_DIR, "datasetcovid")
#     ]

#     from src.models.transformer import MultiTaskTransUNet
#     model_class = MultiTaskTransUNet

#     train(args.epochs, model_class, dataset_roots, use_best_config=not args.no_best_config)
