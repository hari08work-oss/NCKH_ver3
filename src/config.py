import torch

# Đường dẫn
DATA_DIR = "data/dataset_split"
MASK_DIR = "data/dataset_split_masks"
CHECKPOINT_DIR = "outputs/checkpoints"

# Cấu hình train
BATCH_SIZE = 4
IMG_SIZE = 288
LR = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
