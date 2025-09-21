import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uuid
import json
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# ==============================
# Config Flask
# ==============================
app = Flask(__name__)
UPLOAD_FOLDER = "web/static/uploads"
RESULT_FOLDER = "web/static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==============================
# Load Model (Transformer đã train)
# ==============================
from src.models.transformer import TransUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# đọc best_config.json để khởi tạo đúng tham số
with open("outputs/best_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)["config"]

model = TransUNet(
    in_channels=1,
    out_channels=1,
    img_size=cfg["img_size"],
    patch_size=cfg["patch_size"],
    emb_size=cfg["emb_size"],
    depth=cfg["depth"],
).to(device)

# load trọng số đã train
checkpoint = torch.load("outputs/checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])   # ✅ load đúng phần model_state
model.eval()

# ==============================
# Hàm dự đoán
# ==============================
def predict_mask(img_path):
    # đọc ảnh grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Không đọc được ảnh: {img_path}")

    # resize về đúng kích thước model yêu cầu
    img_resized = cv2.resize(img, (cfg["img_size"], cfg["img_size"]))
    img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float().cpu().numpy()[0, 0, :, :]

    # resize lại về size gốc ảnh
    mask = cv2.resize(pred, (img.shape[1], img.shape[0]))
    mask = (mask * 255).astype(np.uint8)

    # overlay mask (màu đỏ)
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[mask > 127] = (0, 0, 255)

    return overlay, mask

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            unique_name = str(uuid.uuid4()) + "_" + filename
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(img_path)

            # chạy predict
            overlay, mask = predict_mask(img_path)

            # lưu kết quả
            result_name = "result_" + unique_name
            result_path = os.path.join(RESULT_FOLDER, result_name)
            cv2.imwrite(result_path, overlay)

            return render_template(
                "result.html",
                input_image=url_for("static", filename="uploads/" + unique_name),
                result_image=url_for("static", filename="results/" + result_name),
            )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
