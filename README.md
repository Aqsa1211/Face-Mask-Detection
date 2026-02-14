# 🛡️ Face Mask Detection using MobileNetV2

![Python](https://img.shields.io/badge/Python-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-Educational-blue)

---

## 📌 Project Overview

This project implements a **real-time Face Mask Detection System** using deep learning and computer vision techniques. The system classifies faces into two categories:

* ✅ **With Mask**
* ❌ **Without Mask**

The model is built using transfer learning with **MobileNetV2**, making it lightweight, fast, and suitable for real-time deployment.

---

## 🎯 Objectives

* Build a binary image classifier (Mask / No Mask)
* Apply transfer learning using MobileNetV2
* Use data augmentation for better generalization
* Deploy the model for real-time video detection
* Evaluate performance using standard metrics

---

## 🧠 Model Architecture

### 🔹 Base Model

* **MobileNetV2** (Pre-trained on ImageNet)
* `include_top=False`
* Frozen base layers during initial training

### 🔹 Custom Head

* AveragePooling2D
* Flatten
* Dense (128, ReLU)
* Dropout (0.5)
* Dense (2, Softmax)

---

## 📊 Dataset

* **Source:** Kaggle Face Mask Dataset
* Total Images: ~3,833
* With Mask: 1,915
* Without Mask: 1,918
* Image Size: 224x224

The dataset is balanced and includes various lighting conditions, angles, and backgrounds.

---

## ⚙️ Preprocessing Pipeline

### 🖼 Image Processing

* Resize to 224x224
* Convert to NumPy arrays
* Normalize using MobileNetV2 `preprocess_input()`

### 🏷 Label Encoding

* Binary encoding
* One-hot encoding for softmax

### 🔄 Data Augmentation

* Rotation (±20°)
* Zoom (15%)
* Width & Height Shift (20%)
* Shear (15%)
* Horizontal Flip
* Fill Mode: Nearest

---

## 🧪 Training Configuration

| Parameter        | Value               |
| ---------------- | ------------------- |
| Batch Size       | 32                  |
| Epochs           | 20                  |
| Optimizer        | Adam                |
| Learning Rate    | 1e-4                |
| Loss Function    | Binary Crossentropy |
| Train/Test Split | 80/20               |

---

## 📈 Results

* ✅ Training Accuracy: >95%
* ✅ Validation Accuracy: ~90–95%
* ✅ Minimal Overfitting
* ✅ Strong Generalization

---

## 🎥 Real-Time Detection

The system:

1. Detects faces using OpenCV DNN
2. Extracts face regions
3. Preprocesses each face
4. Predicts mask status
5. Draws bounding boxes:

   * 🟢 Green → Mask
   * 🔴 Red → No Mask

Press **Q** to exit the video stream.

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
python train_mask_detector.py
```

### 4️⃣ Run Real-Time Detection

```bash
python detect_mask_video.py
```

---

## 📂 Project Structure

```
├── dataset/
├── face_detector/
├── detect_mask_video.py
├── train_mask_detector.py
├── requirements.txt
├── mask_detector.model
└── README.md
```

---


