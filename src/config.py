import torch

# Đường dẫn
DATA_DIR = "data"
CHECKPOINT_DIR = "outputs/checkpoints"

# Cấu hình train
BATCH_SIZE = 4
IMG_SIZE = 288
LR = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
