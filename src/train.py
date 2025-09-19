import argparse, json, time, os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import *
from src.data_loader import get_loader, get_transforms
from src.utils import dice_score, iou_score, save_checkpoint


def train(epochs, model_class, use_best_config=True, best_config_path="outputs/best_config.json"):
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
    criterion = nn.BCEWithLogitsLoss()

    # ----- transforms -----
    train_tf = get_transforms(cfg["img_size"], is_train=True)
    val_tf   = get_transforms(cfg["img_size"], is_train=False)

    # ----- data -----
    train_loader = get_loader(
        DATA_DIR + "/train", MASK_DIR + "/train",
        batch_size=cfg["batch_size"], img_size=cfg["img_size"],
        transform=train_tf
    )
    val_loader = get_loader(
        DATA_DIR + "/val", MASK_DIR + "/val",
        batch_size=cfg["batch_size"], shuffle=False, img_size=cfg["img_size"],
        transform=val_tf
    )

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

            # đảm bảo masks có shape [N,1,H,W]
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

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
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        avg_iou = total_iou / len(train_loader)

        # ----- validation -----
        model.eval()
        val_loss, val_dice, val_iou = 0, 0, 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Số epoch huấn luyện")
    parser.add_argument("--no_best_config", action="store_true", help="Không dùng best_config.json")
    parser.add_argument("--model", type=str, default="transformer", choices=["unet", "transformer"], help="Chọn mô hình train")

    args = parser.parse_args()

    if args.model == "unet":
        print("[INFO] Training baseline UNet...")
        from src.models.unet import UNet
        model_class = UNet
    else:
        print("[INFO] Training Transformer (TransUNet)...")
        from src.models.transformer import TransUNet
        model_class = TransUNet

    train(args.epochs, model_class, use_best_config=not args.no_best_config)
