# src/validate.py
import os
import cv2
import torch
import torch.nn as nn
from src.config import *
from src.data_loader import get_loader
from src.models.unet import UNet      # hoặc TransUNet nếu bạn muốn
from src.utils import dice_score, iou_score

# thư mục lưu kết quả trực quan
RESULT_DIR = "outputs/val_results"
os.makedirs(RESULT_DIR, exist_ok=True)

def validate():
    # ----- load model -----
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)  # đổi sang TransUNet nếu cần
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pth", map_location=DEVICE))
    model.eval()

    # ----- data loader -----
    val_loader = get_loader(DATA_DIR + "/val", MASK_DIR + "/val", 
                            batch_size=BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)

    criterion = nn.BCEWithLogitsLoss()
    total_loss, total_dice, total_iou = 0, 0, 0

    with torch.no_grad():
        for i, (imgs, masks, _) in enumerate(val_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)

            # resize nếu output khác size mask
            if preds.shape[2:] != masks.shape[2:]:
                preds = torch.nn.functional.interpolate(
                    preds, size=masks.shape[2:], mode="bilinear", align_corners=False
                )

            loss = criterion(preds, masks)
            total_loss += loss.item()
            total_dice += dice_score(preds, masks).item()
            total_iou  += iou_score(preds, masks).item()

            # ---- lưu mask + bbox ----
            probs = torch.sigmoid(preds)
            pred_mask = (probs > 0.5).float()[0, 0].cpu().numpy() * 255
            gt_mask   = masks[0, 0].cpu().numpy() * 255

            # tìm bbox
            contours, _ = cv2.findContours(pred_mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vis_img = (imgs[0, 0].cpu().numpy() * 255).astype("uint8")
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)  # bbox đỏ

            # ghép ground-truth mask và pred mask
            out_img = cv2.hconcat([
                cv2.cvtColor(gt_mask.astype("uint8"), cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(pred_mask.astype("uint8"), cv2.COLOR_GRAY2BGR),
                vis_img
            ])

            cv2.imwrite(f"{RESULT_DIR}/val_{i}.png", out_img)

    # ---- in kết quả ----
    print(f"[VAL] Loss={total_loss/len(val_loader):.4f}, "
          f"Dice={total_dice/len(val_loader):.4f}, "
          f"IoU={total_iou/len(val_loader):.4f}")

if __name__ == "__main__":
    validate()
