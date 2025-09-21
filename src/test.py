import os
import cv2
import torch
from src.config import *
from src.data_loader import get_loader
from src.models.unet import UNet
from src.utils import dice_score, iou_score

def test():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/model_epoch{EPOCHS}.pth", map_location=DEVICE))
    model.eval()

    test_loader = get_loader(DATA_DIR+"/test", MASK_DIR+"/test", BATCH_SIZE, shuffle=False)

    total_dice, total_iou = 0, 0
    os.makedirs("outputs/results", exist_ok=True)

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(test_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)

            total_dice += dice_score(preds, masks).item()
            total_iou += iou_score(preds, masks).item()

            # lưu ảnh kết quả
            pred_mask = torch.sigmoid(preds)
            pred_mask = (pred_mask > 0.5).float().cpu().numpy()[0,0,:,:]*255
            img = imgs[0].cpu().numpy()[0]*255
            mask = masks[0].cpu().numpy()[0]*255

            cv2.imwrite(f"outputs/results/{idx}_img.png", img)
            cv2.imwrite(f"outputs/results/{idx}_gt.png", mask)
            cv2.imwrite(f"outputs/results/{idx}_pred.png", pred_mask)

    print(f"[TEST] Dice: {total_dice/len(test_loader):.4f}, IoU: {total_iou/len(test_loader):.4f}")
    print(f"[INFO] Kết quả lưu tại outputs/results/")

if __name__ == "__main__":
    test()
