# 🩺 MFWOA + Transformer cho phân đoạn ảnh X-quang/CT

## 📌 Giới thiệu
Dự án này nghiên cứu ứng dụng **Multifactorial Whale Optimization Algorithm (MFWOA)** để tối ưu mô hình **Transformer/UNet** trong bài toán **phân đoạn ảnh y tế (X-quang/CT)**.  
Kết quả triển khai dưới dạng **Flask Web App**, cho phép người dùng upload ảnh và nhận lại kết quả phân đoạn.

---

## 🎯 Mục tiêu
- Nghiên cứu MFWOA và áp dụng để tối ưu siêu tham số huấn luyện.  
- Xây dựng pipeline phân đoạn ảnh bằng **UNet, TransUNet, Hybrid (UNet+Transformer)**.  
- Đánh giá hiệu quả bằng Dice, IoU.  
- Tích hợp demo Flask Web App.  

---

## 📂 Cấu trúc thư mục
```bash
MFWOA_Transformer_Project/
│── README.md
│── requirements.txt
│
├── data/
│ ├── train/ # train images (covid/, noncovid/, …)
│ ├── val/
│ └── test/
│
├── outputs/
│ ├── checkpoints/ # best_model.pth (Transformer), unet_baseline.pth
│ ├── logs/
│ └── results/
│
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── utils.py
│
│ ├── models/
│ │ ├── unet.py # baseline
│ │ ├── transformer.py # Transformer (TransUNet/SwinUNet) (s dụng mô hình transUnet )
│ │ └── multitask.py # optional (segmentation + classification)
│
│ ├── optimization/
│ │ ├── mfwoa.py
│ │ └── tune_with_mfwoa.py
│
│ ├── train.py # train final
│ ├── validate.py
│ └── test.py
│
├── web/
│ ├── app.py # Flask backend
│ ├── templates/
│ │ ├── index.html
│ │ └── result.html
│ └── static/
│ ├── uploads/
│ └── results/


## ⚙️ Cài đặt môi trường

1. Clone repo hoặc copy source code vào máy.  
2. Tạo môi trường ảo (Python >= 3.9):  
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # Linux/Mac

3. pip install -r requirements.txt

4. python -m pip install --upgrade pip



## 🚀 Huấn luyện mô hình

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Lưu ý : chuyển sang gpu : set CUDA_VISIBLE_DEVICES=0 (0 or 1 theo gpu của máy )

1. Chạy baseline UNet
python -m src.train --model unet --epochs 5

2. Tìm siêu tham số tốt nhất cho Transformer bằng MFWOA (Tìm kiếm tham số tốt nhất để áp dụng vào thuật toán transformer)

python -m src.optimization.tune_with_mfwoa --iters 20 --pop 12 --proxy_epochs 5 --subset 0.25

-> chạy cho gpu yếu python -m src.optimization.tune_with_mfwoa --iters 20 --pop 12 --proxy_epochs 5 --subset 0.25 --batch_size 1


Kết quả lưu tại: outputs/best_config.json.

3. Train Transformer với config tối ưu
python src/train.py --model transformer --epochs 100 (LƯU Ý : chạy ít epoch để tối ưu trước)


Checkpoint lưu tại outputs/checkpoints/best_model.pth.

##🧪 Validation & Test

1. Validation: python src/validate.py --model transformer

2. Test (lưu overlay kết quả trong outputs/results/): 
-> python src/test.py --model transformer

##🌐 Flask Web Demo

1. Đảm bảo có outputs/checkpoints/best_model.pth.

2. Chạy Flask App:
      python web/app.py

3. Mở trình duyệt: http://127.0.0.1:5000

4. Upload ảnh X-quang/CT → hệ thống hiển thị mask + loại bệnh dự đoán (nếu dùng multi-task).
