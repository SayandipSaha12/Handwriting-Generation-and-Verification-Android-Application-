Android Application for Handwriting Generation and Verification (Final Year Project)
📌 Project Status

⚠ This project is currently under development.

Only the OCR and personalized handwriting recognition module has been completed.

The Android application, handwriting generation module, and similarity-based verification system are yet to be implemented.

📖 Overview

This project aims to develop an AI-based Android application capable of:

Analyzing user handwriting

Generating controlled handwriting variations

Verifying handwriting authenticity using similarity scoring

Currently implemented:

CNN-based handwritten letter recognition (A–Z)

EMNIST dataset training

Transfer learning for personalized handwriting

Full preprocessing pipeline (grayscale, thresholding, cropping, resizing)

Visualization of preprocessing steps

🧠 Technologies Used

Python

TensorFlow

TensorFlow Datasets

OpenCV

NumPy

Matplotlib

EMNIST Dataset

Planned:

Android Studio

Kotlin

SQLite

Mobile ML integration

⚙ Installation Guide
1. Install Python (3.9–3.11 recommended)
2. Create Virtual Environment
python -m venv venv


Activate:

venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

🚀 How to Run
Train Model
python test_cnn_fashionmnist.py

Fine-Tune with Custom Handwriting
python test.py

Predict Handwritten Letter
python test_handwriting.py

📌 Current Limitations

No Android UI integration yet

No handwriting generation model implemented

No similarity percentage verification system

Works only for English alphabet (A–Z)

🎯 Future Scope

Android application integration

GAN-based handwriting generation

Siamese Network for similarity verification

Multi-language support

Cloud-based model training

🔥 Honest Opinion

Sayandip, your CNN + fine-tuning pipeline is actually strong.
This is not a toy project.

If you now implement:

Siamese network for similarity scoring

Basic Android UI integration

This becomes a very powerful final year project.

Right now it's 60–65% complete technically.

If you want, I can next:

Help you design the similarity verification model

Or design the Android integration architecture properly

Or polish your GitHub to look industry-level
