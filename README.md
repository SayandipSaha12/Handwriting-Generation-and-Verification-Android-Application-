# Handwriting Generation and Verification Android Application  
*(Final Year Project)*

---

## 📌 Project Status

⚠ This project is currently **under development**.

Only the **OCR and personalized handwriting recognition module** has been completed.

The following modules are **not yet implemented**:

- Android application integration  
- Handwriting generation model  
- Similarity-based handwriting verification system  

---

## 📖 Overview

This project aims to develop an AI-based Android application capable of:

- Analyzing user handwriting  
- Generating controlled handwriting variations  
- Verifying handwriting authenticity using similarity scoring  

### ✅ Currently Implemented

- CNN-based handwritten letter recognition (A–Z)  
- EMNIST dataset training  
- Transfer learning for personalized handwriting  
- Full preprocessing pipeline:
  - Grayscale conversion  
  - Image inversion  
  - Otsu thresholding  
  - Character cropping  
  - Padding and centering  
  - Resizing to 28×28  
- Visualization of preprocessing steps  

---

## 🧠 Technologies Used

### Implemented

- Python  
- TensorFlow  
- TensorFlow Datasets  
- OpenCV  
- NumPy  
- Matplotlib  
- EMNIST Dataset  

### Planned

- Android Studio  
- Kotlin  
- SQLite  
- Mobile ML integration  

---

## ⚙ Installation Guide

### 1️⃣ Install Python

Download and install Python (3.9–3.11 recommended):

https://www.python.org/downloads/

Make sure to check:  
Add Python to PATH

---

### 2️⃣ Create Virtual Environment

Run:

```bash
python -m venv venv
```

Activate virtual environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

Create a file named `requirements.txt` with:

```
tensorflow
tensorflow-datasets
opencv-python
numpy
matplotlib
```

Then run:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🔹 Train Base Model (EMNIST)

```bash
python test_cnn_fashionmnist.py
```

This will generate:

- handwriting_emnist_model_improved.keras  
- training_history.png  
- sample_predictions.png  

---

### 🔹 Fine-Tune with Custom Handwriting

Before running, create a folder for your handwritten samples and update the folder path inside `test.py`.

Then run:

```bash
python test.py
```

This will generate:

- handwriting_model_personalized.keras  
- finetuning_history.png  

---

### 🔹 Predict Handwritten Letter

Update the image path inside `test_handwriting.py`, then run:

```bash
python test_handwriting.py
```

This will:

- Predict the letter  
- Show top 5 predictions  
- Display confidence percentage  
- Save preprocessing visualization  

---

## 📌 Current Limitations

- No Android UI integration yet  
- No handwriting generation model implemented  
- No similarity percentage verification system  
- Works only for English alphabet (A–Z)  
- No cloud-based storage or API integration  

---

## 🎯 Future Scope

- Android application integration  
- GAN-based handwriting generation  
- Siamese Network for similarity verification  
- Multi-language support  
- Cloud-based model training  
- Secure local database integration (SQLite)  

---

## 👨‍💻 Developer

Sayandip Saha - 4th Year IT (St. Thomas' College of Engineering and Technology)
Agniva Acherjee - 4th Year IT (St. Thomas' College of Engineering and Technology)
Md Farhann Akhter - 4th Year IT (St. Thomas' College of Engineering and Technology)
