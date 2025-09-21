# import argparse, json, random, os
# from pathlib import Path
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# # --- imports dự án ---
# from src.config import DATA_DIR, DEVICE
# from src.data_loader import get_loader
# try:
#     from src.data_loader import get_transforms
#     HAS_AUG = True
# except Exception:
#     HAS_AUG = False

# from src.models.transformer import TransUNet
# from src.utils import dice_score
# from src.optimization.mfwoa import MFWOA


# # ---------- helpers ----------
# def to_int(x): return int(round(x))
# def clamp(v, lo, hi): return max(lo, min(hi, v))


# def map_vector_to_config(vec):
#     """
#     vec = [lr, wd, depth, heads, emb, patch, loss_w, img_size]
#     Batch size fix = 1 để tránh OOM khi proxy.
#     """
#     lr      = 10 ** np.interp(vec[0], [0, 1], [-5, -3])   # 1e-5 .. 1e-3
#     wd      = 10 ** np.interp(vec[1], [0, 1], [-6, -3])   # 1e-6 .. 1e-3
#     depth   = clamp(to_int(np.interp(vec[2], [0, 1], [2, 6])), 2, 6)
#     heads   = clamp(2 * to_int(np.interp(vec[3], [0, 1], [1, 4])), 2, 8)
#     emb     = [128, 192, 256][clamp(to_int(np.interp(vec[4], [0, 1], [0, 2])), 0, 2)]
#     patch   = [8, 16][clamp(to_int(np.interp(vec[5], [0, 1], [0, 1])), 0, 1)]
#     loss_w  = np.interp(vec[6], [0, 1], [0.0, 1.0])
#     img_sz  = [256, 288][clamp(to_int(np.interp(vec[7], [0, 1], [0, 1])), 0, 1)]

#     return {
#         "lr": float(lr),
#         "weight_decay": float(wd),
#         "depth": int(depth),
#         "num_heads": int(heads),
#         "emb_size": int(emb),
#         "patch_size": int(patch),
#         "loss_cls_weight": float(loss_w),
#         "img_size": int(img_sz),
#         "batch_size": 1
#     }


# def build_model(cfg):
#     return TransUNet(
#         in_channels=1,
#         out_channels=1,
#         img_size=cfg["img_size"],
#         patch_size=cfg["patch_size"],
#         emb_size=cfg["emb_size"],
#         depth=cfg["depth"],
#     )


# def _ensure_nonempty_dir(p: Path, label: str):
#     if not p.exists():
#         raise RuntimeError(f"[MFWOA] {label} không tồn tại: {p}")
#     exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
#     any_file = any(x.suffix.lower() in exts for x in p.rglob("*"))
#     if not any_file:
#         raise RuntimeError(f"[MFWOA] {label} rỗng hoặc không có ảnh {exts}: {p}")


# # ---------- Evaluation (objective function cho MFWOA) ----------
# def evaluate_config(cfg, train_subset_ratio=0.25, proxy_epochs=5, seed=42):
#     """
#     Train proxy và validate → trả về (fitness, dice).
#     MFWOA sẽ minimize fitness = 1 - Dice.
#     Dice chỉ để log, không tham gia tối ưu trực tiếp.
#     """
#     random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

#     device = DEVICE
#     img_sz = cfg["img_size"]

#     # transforms
#     train_tf = get_transforms(img_sz, is_train=True) if HAS_AUG else None
#     val_tf   = get_transforms(img_sz, is_train=False) if HAS_AUG else None

#     # paths
#     data_dir = Path(DATA_DIR)
#     train_img_dir = data_dir / "train" / "images"
#     val_img_dir   = data_dir / "val" / "images"
#     train_msk_dir = data_dir / "train" / "masks"
#     val_msk_dir   = data_dir / "val" / "masks"

#     print(f"[MFWOA] train images: {train_img_dir}")
#     print(f"[MFWOA] train masks : {train_msk_dir}")
#     print(f"[MFWOA] val images  : {val_img_dir}")
#     print(f"[MFWOA] val masks   : {val_msk_dir}")

#     _ensure_nonempty_dir(train_img_dir, "Thư mục ảnh train")
#     _ensure_nonempty_dir(train_msk_dir, "Thư mục mask train")
#     _ensure_nonempty_dir(val_img_dir, "Thư mục ảnh val")
#     _ensure_nonempty_dir(val_msk_dir, "Thư mục mask val")

#     # loaders
#     full_train = get_loader(train_img_dir, train_msk_dir,
#                             batch_size=cfg["batch_size"], shuffle=True,
#                             img_size=img_sz, transform=train_tf)
#     val_loader = get_loader(val_img_dir, val_msk_dir,
#                             batch_size=cfg["batch_size"], shuffle=False,
#                             img_size=img_sz, transform=val_tf)

#     print(f"[INFO] Loaded {len(full_train.dataset)} train samples, {len(val_loader.dataset)} val samples.")

#     # lấy subset batch cho proxy
#     n_batches = max(1, int(len(full_train) * train_subset_ratio))
#     sub_batches = []
#     for i, batch in enumerate(full_train):
#         sub_batches.append(batch)
#         if i + 1 >= n_batches: break

#     # model
#     model = build_model(cfg).to(device)
#     opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
#     criterion = nn.BCEWithLogitsLoss()

#     # proxy train
#     model.train()
#     for ep in range(proxy_epochs):
#         ep_loss = 0.0
#         for b, (imgs, masks, _) in enumerate(sub_batches):
#             imgs, masks = imgs.to(device), masks.to(device)
#             preds = model(imgs)

#             if preds.shape[2:] != masks.shape[2:]:
#                 preds = torch.nn.functional.interpolate(
#                     preds, size=masks.shape[2:], mode="bilinear", align_corners=False
#                 )

#             loss = criterion(preds, masks)
#             opt.zero_grad(); loss.backward(); opt.step()
#             ep_loss += loss.item()

#         print(f"[Proxy] Epoch {ep+1}/{proxy_epochs} AVG Loss={ep_loss/len(sub_batches):.4f}")

#     # validate
#     model.eval()
#     total_dice = 0.0
#     with torch.no_grad():
#         for imgs, masks, _ in val_loader:
#             imgs, masks = imgs.to(device), masks.to(device)
#             preds = model(imgs)
#             if preds.shape[2:] != masks.shape[2:]:
#                 preds = torch.nn.functional.interpolate(
#                     preds, size=masks.shape[2:], mode="bilinear", align_corners=False
#                 )
#             total_dice += dice_score(preds, masks).item()

#     avg_dice = total_dice / max(1, len(val_loader))
#     fitness = 1.0 - avg_dice  # cái mà MFWOA sẽ minimize
#     return fitness, avg_dice


# # ---------- MFWOA Runner ----------
# def run_mfwoa(iters=20, pop=12, subset_ratio=0.25, proxy_epochs=5,
#               seed=42, save_path="outputs/best_config.json"):
#     dim = 8
#     bounds = (0.0, 1.0)

#     history = []
#     call_counter = {"count": 0}

#     def obj_func(vec):
#         call_counter["count"] += 1
#         eval_id = call_counter["count"]
#         iter_idx = (eval_id - 1) // pop + 1
#         indiv_idx = (eval_id - 1) % pop + 1

#         cfg = map_vector_to_config(vec)
#         fitness, dice = evaluate_config(cfg, train_subset_ratio=subset_ratio,
#                                         proxy_epochs=proxy_epochs, seed=seed)
#         history.append((eval_id, fitness, dice))

#         print(f"[MFWOA] Iter {iter_idx}/{iters} | Individual {indiv_idx}/{pop} "
#               f"| Dice={dice:.4f} | Fitness={fitness:.4f}")
#         return fitness

#     opt = MFWOA(obj_func=obj_func, dim=dim, bounds=bounds,
#                 pop_size=pop, max_iter=iters)

#     best_vec, best_obj = opt.optimize()
#     best_cfg = map_vector_to_config(best_vec)

#     _, best_dice = evaluate_config(best_cfg, train_subset_ratio=subset_ratio,
#                                    proxy_epochs=proxy_epochs, seed=seed)

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump({"config": best_cfg, "val_dice_proxy": best_dice},
#                   f, ensure_ascii=False, indent=2)

#     print("[MFWOA] Best proxy Dice =", f"{best_dice:.4f}")
#     print("[MFWOA] Saved best_config to", save_path)

#     # vẽ biểu đồ
#     if history:
#         eval_ids = [h[0] for h in history]
#         fitnesses = [h[1] for h in history]
#         dices = [h[2] for h in history]

#         plt.figure(figsize=(10,5))
#         plt.plot(eval_ids, fitnesses, label="Fitness (1 - Dice)", color="red")
#         plt.plot(eval_ids, dices, label="Dice", color="blue")
#         plt.xlabel("Evaluations")
#         plt.ylabel("Value")
#         plt.title("MFWOA Optimization Progress")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig("outputs/mfwoa_progress.png")
#         plt.show()

#     return best_cfg


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--iters", type=int, default=20, help="Số vòng lặp MFWOA")
#     parser.add_argument("--pop", type=int, default=12, help="Kích thước quần thể")
#     parser.add_argument("--subset", type=float, default=0.25, help="Tỉ lệ subset train (0..1)")
#     parser.add_argument("--proxy_epochs", type=int, default=5, help="Số epoch proxy mỗi đánh giá")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--save", type=str, default="outputs/best_config.json")
#     args = parser.parse_args()

#     run_mfwoa(iters=args.iters, pop=args.pop, subset_ratio=args.subset,
#               proxy_epochs=args.proxy_epochs, seed=args.seed, save_path=args.save)




#ver_2 

import argparse, json, random, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt

# --- dự án ---
from src.config import DEVICE
from src.data_loader import get_loader
try:
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
    Batch size cố định = 1 cho proxy (tránh OOM).
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
        "batch_size": 1
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

def _ensure_nonempty_dir(p: Path, label: str):
    if not p.exists():
        raise RuntimeError(f"[MFWOA] {label} không tồn tại: {p}")
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    any_file = any(x.suffix.lower() in exts for x in p.rglob("*"))
    if not any_file:
        raise RuntimeError(f"[MFWOA] {label} rỗng hoặc không có ảnh {exts}: {p}")

def _safe_unpack(batch):
    """Hỗ trợ batch dạng (img, mask) hoặc (img, mask, label)."""
    if len(batch) == 2:
        imgs, masks = batch
        return imgs, masks
    elif len(batch) == 3:
        imgs, masks, _ = batch
        return imgs, masks
    else:
        raise RuntimeError(f"[MFWOA] Batch có {len(batch)} phần tử không hợp lệ.")

# ---------- gộp nhiều dataset ----------
def get_multi_loaders(all_base_dirs, cfg):
    """
    Nhận danh sách đường dẫn dataset (mỗi dataset có train/images,masks & val/images,masks)
    → Trả về train_loader, val_loader đã gộp.
    """
    img_sz = cfg["img_size"]
    train_tf = get_transforms(img_sz, is_train=True) if HAS_AUG else None
    val_tf   = get_transforms(img_sz, is_train=False) if HAS_AUG else None

    train_datasets, val_datasets = [], []
    total_tr, total_vl = 0, 0

    for base in all_base_dirs:
        base = Path(base)
        train_img = base / "train" / "images"
        train_msk = base / "train" / "masks"
        val_img   = base / "val" / "images"
        val_msk   = base / "val" / "masks"

        if not (train_img.exists() and val_img.exists()):
            print(f"[WARN] Bỏ qua {base} (thiếu train/val).")
            continue

        # kiểm tra có file
        _ensure_nonempty_dir(train_img, f"Ảnh train ({base.name})")
        _ensure_nonempty_dir(train_msk, f"Mask train ({base.name})")
        _ensure_nonempty_dir(val_img,   f"Ảnh val ({base.name})")
        _ensure_nonempty_dir(val_msk,   f"Mask val ({base.name})")

        # tạo loader tạm để lấy dataset/đếm
        tr_loader = get_loader(
            train_img, train_msk,
            batch_size=cfg["batch_size"], shuffle=True,
            img_size=img_sz, transform=train_tf
        )
        vl_loader = get_loader(
            val_img, val_msk,
            batch_size=cfg["batch_size"], shuffle=False,
            img_size=img_sz, transform=val_tf
        )

        train_datasets.append(tr_loader.dataset)
        val_datasets.append(vl_loader.dataset)
        total_tr += len(tr_loader.dataset)
        total_vl += len(vl_loader.dataset)

        print(f"[MFWOA] + Dùng dataset: {base} | train={len(tr_loader.dataset)} | val={len(vl_loader.dataset)}")

    if not train_datasets or not val_datasets:
        raise RuntimeError("[MFWOA] Không tìm thấy dataset hợp lệ trong 'data/'.")

    concat_train = ConcatDataset(train_datasets)
    concat_val   = ConcatDataset(val_datasets)

    print(f"[MFWOA] Tổng sau khi gộp: train={total_tr} | val={total_vl}")

    train_loader = DataLoader(concat_train, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(concat_val,   batch_size=cfg["batch_size"], shuffle=False)
    return train_loader, val_loader

# ---------- đánh giá nhanh 1 cấu hình ----------
def quick_eval(cfg, train_subset_ratio=0.25, proxy_epochs=5, seed=42):
    # seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    device = DEVICE

    # liệt kê các dataset con trong 'data/'
    data_root = Path("data")
    base_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    print("[MFWOA] Tìm thấy các dataset:", [p.name for p in base_dirs])

    # loaders gộp
    full_train, val_loader = get_multi_loaders(base_dirs, cfg)

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

    # proxy train
    model.train()
    for ep in range(proxy_epochs):
        ep_loss = 0.0
        for b, batch in enumerate(sub_batches):
            imgs, masks = _safe_unpack(batch)
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            if preds.shape[2:] != masks.shape[2:]:
                preds = torch.nn.functional.interpolate(
                    preds, size=masks.shape[2:], mode="bilinear", align_corners=False
                )
            loss = criterion(preds, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
            
        print(f"[Proxy] Epoch {ep+1}/{proxy_epochs} | AVG Loss={ep_loss/len(sub_batches):.4f}")

    # validate Dice
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        for batch in val_loader:
            imgs, masks = _safe_unpack(batch)
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            if preds.shape[2:] != masks.shape[2:]:
                preds = torch.nn.functional.interpolate(
                    preds, size=masks.shape[2:], mode="bilinear", align_corners=False
                )
            total_dice += dice_score(preds, masks).item()

    avg_dice = total_dice / max(1, len(val_loader))
    return 1.0 - avg_dice, avg_dice  # MFWOA minimize obj


# ---------- vòng MFWOA ----------
def run_mfwoa(iters, pop, subset_ratio, proxy_epochs,
              seed=42, save_path="outputs/best_config.json"):
    dim = 8
    bounds = (0.0, 1.0)

    history = []
    call_counter = {"count": 0}

    def obj_func(vec):
        call_counter["count"] += 1
        eval_id = call_counter["count"]
        iter_idx  = (eval_id - 1) // pop + 1
        indiv_idx = (eval_id - 1) % pop + 1

        cfg = map_vector_to_config(vec)
        obj, dice = quick_eval(cfg, train_subset_ratio=subset_ratio,
                               proxy_epochs=proxy_epochs, seed=seed)
        history.append((eval_id, obj, dice))

        print(f"[MFWOA] Iter {iter_idx}/{iters} | Individual {indiv_idx}/{pop} "
              f"| Dice={dice:.4f} | Obj={obj:.4f}")
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

    # vẽ tiến trình tối ưu
    if history:
        eval_ids = [h[0] for h in history]
        fitness  = [h[1] for h in history]
        dices    = [h[2] for h in history]

        plt.figure(figsize=(10,5))
        plt.plot(eval_ids, fitness, label="Fitness (1 - Dice)")
        plt.plot(eval_ids, dices,   label="Dice")
        plt.xlabel("Evaluations")
        plt.ylabel("Value")
        plt.title("MFWOA Optimization Progress (multi-dataset)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/mfwoa_progress.png")
        plt.show()

    return best_cfg


# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--iters", type=int, required=True, help="Số vòng lặp MFWOA")
    parser.add_argument("--pop", type=int, required=True, help="Kích thước quần thể")
    parser.add_argument("--subset", type=float, required=True, help="Tỉ lệ subset train (0..1)")
    parser.add_argument("--proxy_epochs", type=int, required=True, help="Số epoch proxy mỗi đánh giá")
    parser.add_argument("--seed", type=int, required=True, help="Seed ngẫu nhiên")
    parser.add_argument("--save", type=str, required=True, help="Đường dẫn lưu best_config.json")

    args = parser.parse_args()

    run_mfwoa(
        iters=args.iters,
        pop=args.pop,
        subset_ratio=args.subset,
        proxy_epochs=args.proxy_epochs,
        seed=args.seed,
        save_path=args.save
    )





# ##ver_3 chạy nhiều dataset đánh label

# import argparse, json, random, os
# from pathlib import Path
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import ConcatDataset, DataLoader
# import matplotlib.pyplot as plt

# # --- dự án ---
# from src.config import DEVICE
# from src.data_loader import get_loader
# try:
#     from src.data_loader import get_transforms
#     HAS_AUG = True
# except Exception:
#     HAS_AUG = False

# from src.models.transformer import MultiTaskTransUNet as TransUNet
# from src.utils import dice_score
# from src.optimization.mfwoa import MFWOA


# # ---------------- Helpers ----------------
# def to_int(x): return int(round(x))
# def clamp(v, lo, hi): return max(lo, min(hi, v))

# def map_vector_to_config(vec):
#     """
#     vec = [lr, wd, depth, heads, emb, patch, loss_w, img_size]
#     """
#     lr      = 10 ** np.interp(vec[0], [0, 1], [-5, -3])   # 1e-5 .. 1e-3
#     wd      = 10 ** np.interp(vec[1], [0, 1], [-6, -3])   # 1e-6 .. 1e-3
#     depth   = clamp(to_int(np.interp(vec[2], [0, 1], [2, 6])), 2, 6)
#     heads   = clamp(2 * to_int(np.interp(vec[3], [0, 1], [1, 4])), 2, 8)
#     emb     = [128, 192, 256][clamp(to_int(np.interp(vec[4], [0, 1], [0, 2])), 0, 2)]
#     patch   = [8, 16][clamp(to_int(np.interp(vec[5], [0, 1], [0, 1])), 0, 1)]
#     loss_w  = np.interp(vec[6], [0, 1], [0.0, 1.0])
#     img_sz  = [256, 288][clamp(to_int(np.interp(vec[7], [0, 1], [0, 1])), 0, 1)]

#     return {
#         "lr": float(lr),
#         "weight_decay": float(wd),
#         "depth": int(depth),
#         "num_heads": int(heads),
#         "emb_size": int(emb),
#         "patch_size": int(patch),
#         "loss_cls_weight": float(loss_w),
#         "img_size": int(img_sz),
#         "batch_size": 1   # giữ nhỏ để proxy nhanh, tránh OOM
#     }

# def build_model(cfg):
#     return TransUNet(
#         in_channels=1,
#         out_channels=1,
#         img_size=cfg["img_size"],
#         patch_size=cfg["patch_size"],
#         emb_size=cfg["emb_size"],
#         depth=cfg["depth"],
#     )

# def _ensure_nonempty_dir(p: Path, label: str):
#     if not p.exists():
#         raise RuntimeError(f"[MFWOA] {label} không tồn tại: {p}")
#     exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
#     if not any(x.suffix.lower() in exts for x in p.rglob("*")):
#         raise RuntimeError(f"[MFWOA] {label} rỗng hoặc không có ảnh {exts}: {p}")

# def _safe_unpack(batch):
#     """Trích xuất imgs, masks từ batch (img, mask) hoặc (img, mask, label)."""
#     if len(batch) == 2:
#         return batch[0], batch[1]
#     elif len(batch) == 3:
#         return batch[0], batch[1]
#     else:
#         raise RuntimeError(f"[MFWOA] Batch có {len(batch)} phần tử không hợp lệ.")


# # ---------------- Multi-dataset Loader ----------------
# def get_multi_loaders(all_base_dirs, cfg):
#     img_sz = cfg["img_size"]
#     train_tf = get_transforms(img_sz, is_train=True) if HAS_AUG else None
#     val_tf   = get_transforms(img_sz, is_train=False) if HAS_AUG else None

#     train_datasets, val_datasets = [], []
#     total_tr, total_vl = 0, 0

#     for base in all_base_dirs:
#         base = Path(base)
#         train_img = base / "train" / "images"
#         train_msk = base / "train" / "masks"
#         val_img   = base / "val" / "images"
#         val_msk   = base / "val" / "masks"

#         if not (train_img.exists() and val_img.exists()):
#             print(f"[WARN] Bỏ qua {base} (thiếu train/val).")
#             continue

#         _ensure_nonempty_dir(train_img, f"Ảnh train ({base.name})")
#         _ensure_nonempty_dir(train_msk, f"Mask train ({base.name})")
#         _ensure_nonempty_dir(val_img,   f"Ảnh val ({base.name})")
#         _ensure_nonempty_dir(val_msk,   f"Mask val ({base.name})")

#         tr_loader = get_loader(train_img, train_msk, batch_size=cfg["batch_size"],
#                                shuffle=True, img_size=img_sz, transform=train_tf)
#         vl_loader = get_loader(val_img, val_msk, batch_size=cfg["batch_size"],
#                                shuffle=False, img_size=img_sz, transform=val_tf)

#         train_datasets.append(tr_loader.dataset)
#         val_datasets.append(vl_loader.dataset)
#         total_tr += len(tr_loader.dataset)
#         total_vl += len(vl_loader.dataset)

#         print(f"[MFWOA] + {base.name}: train={len(tr_loader.dataset)} | val={len(vl_loader.dataset)}")

#     if not train_datasets or not val_datasets:
#         raise RuntimeError("[MFWOA] Không tìm thấy dataset hợp lệ trong 'data/'.")

#     concat_train = ConcatDataset(train_datasets)
#     concat_val   = ConcatDataset(val_datasets)

#     print(f"[MFWOA] Tổng sau khi gộp: train={total_tr} | val={total_vl}")

#     train_loader = DataLoader(concat_train, batch_size=cfg["batch_size"], shuffle=True)
#     val_loader   = DataLoader(concat_val,   batch_size=cfg["batch_size"], shuffle=False)
#     return train_loader, val_loader


# # ---------------- Quick Evaluation ----------------
# def quick_eval(cfg, train_subset_ratio=0.25, proxy_epochs=5, seed=42):
#     random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

#     device = DEVICE
#     data_root = Path("data")
#     base_dirs = [p for p in data_root.iterdir() if p.is_dir()]
#     print("[MFWOA] Tìm thấy dataset:", [p.name for p in base_dirs])

#     full_train, val_loader = get_multi_loaders(base_dirs, cfg)

#     # subset proxy
#     n_batches = max(1, int(len(full_train) * train_subset_ratio))
#     sub_batches = []
#     for i, batch in enumerate(full_train):
#         sub_batches.append(batch)
#         if i + 1 >= n_batches: break

#     model = build_model(cfg).to(device)
#     opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
#     criterion = nn.BCEWithLogitsLoss()

#     # proxy train
#     model.train()
#     for ep in range(proxy_epochs):
#         ep_loss = 0.0
#         for batch in sub_batches:
#             imgs, masks = _safe_unpack(batch)
#             imgs, masks = imgs.to(device), masks.to(device)
#             preds = model(imgs)
#             if preds.shape[2:] != masks.shape[2:]:
#                 preds = torch.nn.functional.interpolate(preds, size=masks.shape[2:],
#                                                         mode="bilinear", align_corners=False)
#             loss = criterion(preds, masks)
#             opt.zero_grad(); loss.backward(); opt.step()
#             ep_loss += loss.item()
#         print(f"[Proxy] Epoch {ep+1}/{proxy_epochs} | AVG Loss={ep_loss/len(sub_batches):.4f}")

#     # validation (Dice)
#     model.eval()
#     total_dice = 0.0
#     with torch.no_grad():
#         for batch in val_loader:
#             imgs, masks = _safe_unpack(batch)
#             imgs, masks = imgs.to(device), masks.to(device)
#             preds = model(imgs)
#             if preds.shape[2:] != masks.shape[2:]:
#                 preds = torch.nn.functional.interpolate(preds, size=masks.shape[2:],
#                                                         mode="bilinear", align_corners=False)
#             total_dice += dice_score(preds, masks).item()

#     avg_dice = total_dice / max(1, len(val_loader))
#     return 1.0 - avg_dice, avg_dice


# # ---------------- MFWOA Main Loop ----------------
# def run_mfwoa(iters=20, pop=12, subset_ratio=0.25, proxy_epochs=5,
#               seed=42, save_path="outputs/best_config.json"):
#     dim = 8
#     bounds = (0.0, 1.0)
#     history = []
#     call_counter = {"count": 0}

#     def obj_func(vec):
#         call_counter["count"] += 1
#         eval_id = call_counter["count"]
#         iter_idx  = (eval_id - 1) // pop + 1
#         indiv_idx = (eval_id - 1) % pop + 1

#         cfg = map_vector_to_config(vec)
#         obj, dice = quick_eval(cfg, train_subset_ratio=subset_ratio,
#                                proxy_epochs=proxy_epochs, seed=seed)
#         history.append((eval_id, obj, dice))

#         print(f"[MFWOA] Iter {iter_idx}/{iters} | Indiv {indiv_idx}/{pop} "
#               f"| Dice={dice:.4f} | Obj={obj:.4f}")
#         return obj

#     opt = MFWOA(obj_func=obj_func, dim=dim, bounds=bounds,
#                 pop_size=pop, max_iter=iters)

#     best_vec, best_obj = opt.optimize()
#     best_cfg = map_vector_to_config(best_vec)

#     _, best_dice = quick_eval(best_cfg, train_subset_ratio=subset_ratio,
#                               proxy_epochs=proxy_epochs, seed=seed)

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump({"config": best_cfg, "val_dice_proxy": best_dice},
#                   f, ensure_ascii=False, indent=2)

#     print(f"[MFWOA] Best proxy Dice = {best_dice:.4f}")
#     print(f"[MFWOA] Saved config → {save_path}")

#     # plot history
#     if history:
#         eval_ids = [h[0] for h in history]
#         fitness  = [h[1] for h in history]
#         dices    = [h[2] for h in history]

#         plt.figure(figsize=(10,5))
#         plt.plot(eval_ids, fitness, label="Fitness (1 - Dice)", color="red")
#         plt.plot(eval_ids, dices,   label="Dice", color="blue")
#         plt.xlabel("Evaluations"); plt.ylabel("Value")
#         plt.title("MFWOA Optimization Progress (multi-dataset)")
#         plt.legend(); plt.grid(True); plt.tight_layout()
#         os.makedirs("outputs", exist_ok=True)
#         plt.savefig("outputs/mfwoa_progress.png")
#         plt.show()

#     return best_cfg


# # ---------------- main ----------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--iters", type=int, default=20)
#     parser.add_argument("--pop", type=int, default=12)
#     parser.add_argument("--subset", type=float, default=0.25)
#     parser.add_argument("--proxy_epochs", type=int, default=5)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--save", type=str, default="outputs/best_config.json")
#     args = parser.parse_args()

#     run_mfwoa(iters=args.iters, pop=args.pop,
#               subset_ratio=args.subset, proxy_epochs=args.proxy_epochs,
#               seed=args.seed, save_path=args.save)
