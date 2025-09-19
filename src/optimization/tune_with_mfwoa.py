# src/optimization/tune_with_mfwoa.py
import argparse, json, random, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import *
from src.data_loader import get_loader
try:
    # nếu bạn đã thêm augmentation ở data_loader.py
    from src.data_loader import get_transforms
    HAS_AUG = True
except Exception:
    HAS_AUG = False

from src.models.transformer import TransUNet
from src.utils import dice_score
from src.optimization.mfwoa import MFWOA


# ---------- helpers ----------
def to_int(x): return int(round(x))
def clamp(v, lo, hi): return max(lo, min(hi, v))


def map_vector_to_config(vec):
    """
    vec = [lr, wd, depth, heads, emb, patch, loss_w, img_size]
    Batch size fix = 1 để tránh OOM khi proxy.
    """
    lr      = 10 ** np.interp(vec[0], [0, 1], [-5, -3])         # 1e-5 .. 1e-3
    wd      = 10 ** np.interp(vec[1], [0, 1], [-6, -3])         # 1e-6 .. 1e-3
    depth   = clamp(to_int(np.interp(vec[2], [0, 1], [2, 6])), 2, 6)
    heads   = clamp(2 * to_int(np.interp(vec[3], [0, 1], [1, 4])), 2, 8)  # 2,4,6,8
    emb     = [128, 192, 256][clamp(to_int(np.interp(vec[4], [0, 1], [0, 2])), 0, 2)]
    patch   = [8, 16][clamp(to_int(np.interp(vec[5], [0, 1], [0, 1])), 0, 1)]
    loss_w  = np.interp(vec[6], [0, 1], [0.0, 1.0])
    img_sz  = [256, 288][clamp(to_int(np.interp(vec[7], [0, 1], [0, 1])), 0, 1)]

    return {
        "lr": float(lr),
        "weight_decay": float(wd),
        "depth": int(depth),
        "num_heads": int(heads),
        "emb_size": int(emb),
        "patch_size": int(patch),
        "loss_cls_weight": float(loss_w),
        "img_size": int(img_sz),
        "batch_size": 1  # proxy nhanh & tránh OOM
    }


def build_model(cfg):
    return TransUNet(
        in_channels=1,
        out_channels=1,
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        emb_size=cfg["emb_size"],
        depth=cfg["depth"],
    )


def quick_eval(cfg, train_subset_ratio=0.25, proxy_epochs=5, seed=42):
    # seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    device = DEVICE
    img_sz = cfg["img_size"]

    # transforms (nếu có)
    train_tf = get_transforms(img_sz, is_train=True) if HAS_AUG else None
    val_tf   = get_transforms(img_sz, is_train=False) if HAS_AUG else None

    # loaders
    full_train = get_loader(
        DATA_DIR + "/train", MASK_DIR + "/train",
        batch_size=cfg["batch_size"], shuffle=True, img_size=img_sz,
        transform=train_tf
    )
    val_loader = get_loader(
        DATA_DIR + "/val", MASK_DIR + "/val",
        batch_size=cfg["batch_size"], shuffle=False, img_size=img_sz,
        transform=val_tf
    )

    # lấy subset batch cho proxy
    n_batches = max(1, int(len(full_train) * train_subset_ratio))
    sub_batches = []
    for i, batch in enumerate(full_train):
        sub_batches.append(batch)
        if i + 1 >= n_batches:
            break

    model = build_model(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()

    # -------- proxy train --------
    model.train()
    for ep in range(proxy_epochs):
        ep_loss = 0.0
        for b, (imgs, masks, _) in enumerate(sub_batches):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            # đảm bảo cùng kích thước
            if preds.shape[2:] != masks.shape[2:]:
                preds = torch.nn.functional.interpolate(
                    preds, size=masks.shape[2:], mode="bilinear", align_corners=False
                )

            loss = criterion(preds, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()

            print(f"[Proxy] Epoch {ep+1}/{proxy_epochs} | Batch {b+1}/{len(sub_batches)} | Loss={loss.item():.4f}")

        print(f"[Proxy] Epoch {ep+1}/{proxy_epochs} AVG Loss={ep_loss/len(sub_batches):.4f}")

    # -------- validate (Dice) --------
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            if preds.shape[2:] != masks.shape[2:]:
                preds = torch.nn.functional.interpolate(
                    preds, size=masks.shape[2:], mode="bilinear", align_corners=False
                )
            total_dice += dice_score(preds, masks).item()

    avg_dice = total_dice / max(1, len(val_loader))
    # MFWOA minimize → obj = 1 - Dice
    return 1.0 - avg_dice, avg_dice


def run_mfwoa(iters=20, pop=12, subset_ratio=0.25, proxy_epochs=5,
              seed=42, save_path="outputs/best_config.json"):
    dim = 8
    bounds = (0.0, 1.0)

    def obj_func(vec):
        cfg = map_vector_to_config(vec)
        obj, dice = quick_eval(cfg, train_subset_ratio=subset_ratio,
                               proxy_epochs=proxy_epochs, seed=seed)
        # log mỗi cá thể
        print(f"[DEBUG] obj_func called → Dice={dice:.4f}, Obj={obj:.4f}")
        print(f"[DEBUG] Current Config: {cfg}")
        return obj

    opt = MFWOA(obj_func=obj_func, dim=dim, bounds=bounds,
                pop_size=pop, max_iter=iters)

    best_vec, best_obj = opt.optimize()
    best_cfg = map_vector_to_config(best_vec)

    # đánh giá lại best config
    _, best_dice = quick_eval(best_cfg, train_subset_ratio=subset_ratio,
                              proxy_epochs=proxy_epochs, seed=seed)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"config": best_cfg, "val_dice_proxy": best_dice},
                  f, ensure_ascii=False, indent=2)

    print("[MFWOA] Best proxy Dice =", f"{best_dice:.4f}")
    print("[MFWOA] Saved best_config to", save_path)
    return best_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20, help="Số vòng lặp MFWOA")
    parser.add_argument("--pop", type=int, default=12, help="Kích thước quần thể")
    parser.add_argument("--subset", type=float, default=0.25, help="Tỉ lệ subset train (0..1)")
    parser.add_argument("--proxy_epochs", type=int, default=5, help="Số epoch proxy mỗi đánh giá")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="outputs/best_config.json")
    args = parser.parse_args()

    run_mfwoa(iters=args.iters, pop=args.pop, subset_ratio=args.subset,
              proxy_epochs=args.proxy_epochs, seed=args.seed, save_path=args.save)
