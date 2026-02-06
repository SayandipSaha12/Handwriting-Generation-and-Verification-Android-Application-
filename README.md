# ✍️ Handwriting Generation and Verification Android Application
*(AI & ML Based Android Application Development - Final Year Project)*

<div align="center">

[![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-success?style=for-the-badge)](https://github.com/yourusername/handwriting-ai)
[![Python](https://img.shields.io/badge/Python-3.9--3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**AI-Powered Handwriting Recognition, Generation & Verification System**

[Features](#-currently-implemented-phase-1) • [Installation](#-complete-installation-guide) • [Usage](#-how-to-run) • [Tech Stack](#-technologies-used) • [Team](#-project-team)

</div>

---

## 📌 Project Status

> ⚠️ **This project is currently under development.**  
> **Phase 1 (OCR & Recognition)** has been completed successfully.

### ✅ Completed Modules
- CNN-based handwritten letter recognition (A–Z)
- EMNIST dataset training
- Transfer learning for personalized handwriting
- Complete preprocessing pipeline
- Visualization and prediction system

### 🚧 In Progress (Phase 2)
- Android application integration
- Handwriting generation model (GAN-based)
- Similarity-based handwriting verification system
- Multi-language support
- Cloud integration

---

## 📖 Overview

This project aims to develop an **AI-based Android application** capable of:

🎯 **Analyzing user handwriting** - Extract and recognize handwritten characters  
🎯 **Generating controlled handwriting variations** - Create realistic handwriting samples  
🎯 **Verifying handwriting authenticity** - Compare and score handwriting similarity  

### 🎓 Project Goals

| Goal | Description | Status |
|------|-------------|--------|
| **Character Recognition** | Identify handwritten letters A-Z | ✅ Complete |
| **Personalization** | Adapt to individual writing styles | ✅ Complete |
| **Generation** | Create new handwriting samples | 🚧 Planned |
| **Verification** | Authenticate handwriting identity | 🚧 Planned |
| **Mobile App** | Android integration with camera | 🚧 Planned |

---

## ✅ Currently Implemented (Phase 1)

### 🔤 CNN-Based Character Recognition

**Model Architecture:**
- Input: 28×28 grayscale images
- Convolutional layers with MaxPooling
- Dropout for regularization
- Dense layers for classification
- Output: 26 classes (A-Z)

**Training Dataset:**
- **EMNIST** (Extended MNIST) - Balanced split
- Training samples: ~112,800 letters
- Validation samples: ~18,800 letters
- Test accuracy: **~92-95%**

---

### 🎨 Complete Preprocessing Pipeline

Our preprocessing system transforms raw handwriting images into model-ready inputs:

| Step | Operation | Purpose |
|------|-----------|---------|
| 1️⃣ | **Grayscale Conversion** | Reduce complexity, focus on intensity |
| 2️⃣ | **Image Inversion** | White text on black background |
| 3️⃣ | **Otsu Thresholding** | Binary segmentation, noise removal |
| 4️⃣ | **Character Cropping** | Extract bounding box, remove whitespace |
| 5️⃣ | **Padding & Centering** | Add border, center character |
| 6️⃣ | **Resizing** | Normalize to 28×28 pixels |

**Visual Pipeline:**
```
Raw Image → Grayscale → Inverted → Threshold → Cropped → Padded → Resized (28×28)
   📸         🎨          🔄         🔲         ✂️        📏         ✅
```

**Output:** Each step is visualized and saved as `preprocessing_steps.png`

---

### 🎯 Transfer Learning for Personalization

**How it works:**

1. **Base Model**: Pre-trained on EMNIST (general handwriting)
2. **Fine-tuning**: Retrain on user's specific handwriting samples
3. **Adaptation**: Model learns individual writing style
4. **Result**: Personalized recognition with higher accuracy

**Requirements:**
- Minimum 100 handwritten samples per letter
- Clear, well-lit images
- Consistent writing style

**Performance:**
- Base model accuracy: ~92%
- Personalized model accuracy: **~95-98%** (on user's handwriting)

---

### 📊 Visualization Features

**Generated Outputs:**

| File | Content |
|------|---------|
| `training_history.png` | Loss and accuracy curves over epochs |
| `sample_predictions.png` | Model predictions on test samples |
| `finetuning_history.png` | Transfer learning progress |
| `preprocessing_steps.png` | Visual breakdown of image processing |

---

## 🧠 Technologies Used

### ✅ Currently Implemented

<table>
<tr>
<td width="50%">

**Core Technologies**
- **Python 3.9-3.11** - Programming language
- **TensorFlow 2.x** - Deep learning framework
- **TensorFlow Datasets** - EMNIST data loader
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **Matplotlib** - Data visualization

</td>
<td width="50%">

**Dependencies**
```txt
tensorflow>=2.10.0
tensorflow-datasets>=4.8.0
opencv-python>=4.7.0
numpy>=1.23.0
matplotlib>=3.6.0
```

**Dataset**
- **EMNIST Balanced**
  - 26 classes (A-Z uppercase)
  - 28×28 grayscale images
  - ~131,600 training samples

</td>
</tr>
</table>

---

### 🚧 Planned Technologies (Phase 2)

| Technology | Purpose | Status |
|------------|---------|--------|
| **Android Studio** | Mobile app development | 📋 Planned |
| **Kotlin** | Android programming | 📋 Planned |
| **TensorFlow Lite** | Mobile ML inference | 📋 Planned |
| **SQLite** | Local database | 📋 Planned |
| **CameraX API** | Image capture | 📋 Planned |
| **GAN (Generative Adversarial Network)** | Handwriting generation | 📋 Planned |
| **Siamese Network** | Similarity verification | 📋 Planned |

---

## ⚙️ Complete Installation Guide

### 📋 Prerequisites

Before starting, ensure you have:

| Requirement | Version | Download Link | Verify Command |
|------------|---------|---------------|----------------|
| Python | 3.9 - 3.11 | [Download](https://www.python.org/downloads/) | `python --version` |
| pip | Latest | Included with Python | `pip --version` |
| Git | Latest | [Download](https://git-scm.com/downloads) | `git --version` |

**Important:** During Python installation, check ✅ **"Add Python to PATH"**

---

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/handwriting-ai.git
cd handwriting-ai
```

**Or download ZIP:**
1. Go to repository
2. Click "Code" → "Download ZIP"
3. Extract and navigate to folder

---

### 2️⃣ Create Virtual Environment

**Why?** Isolates project dependencies from system Python.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**✅ Success indicator:** Command prompt shows `(venv)` prefix

---

### 3️⃣ Install Dependencies

**Create `requirements.txt`:**
```txt
tensorflow>=2.10.0
tensorflow-datasets>=4.8.0
opencv-python>=4.7.0
numpy>=1.23.0
matplotlib>=3.6.0
Pillow>=9.0.0
```

**Install all packages:**
```bash
pip install -r requirements.txt
```

**⏱️ Installation time:** ~5-10 minutes (depending on internet speed)

**Verify installation:**
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

Expected output: `TensorFlow version: 2.x.x`

---

### 4️⃣ Prepare Directory Structure

Create folders for your handwriting samples:
```bash
mkdir -p handwriting_samples/A
mkdir -p handwriting_samples/B
# ... create folders for all letters A-Z
```

**Or use this script:**

**Windows (PowerShell):**
```powershell
foreach ($letter in 65..90) { New-Item -ItemType Directory -Path "handwriting_samples\$([char]$letter)" -Force }
```

**macOS/Linux:**
```bash
for letter in {A..Z}; do mkdir -p "handwriting_samples/$letter"; done
```

---

## 🚀 How to Run

### 📝 Step 1: Train Base Model (EMNIST)

Train the initial model on the EMNIST dataset:
```bash
python test_cnn_fashionmnist.py
```

**What it does:**
1. Downloads EMNIST dataset (~50MB)
2. Builds CNN architecture
3. Trains for 10 epochs (~5-10 minutes on CPU, ~2-3 minutes on GPU)
4. Saves model as `handwriting_emnist_model_improved.keras`
5. Generates visualization plots

**Generated Files:**
- ✅ `handwriting_emnist_model_improved.keras` - Trained model
- ✅ `training_history.png` - Loss/accuracy curves
- ✅ `sample_predictions.png` - Sample outputs

**Expected Output:**
```
Epoch 1/10
3525/3525 ━━━━━━━━━━━━━━━━━━━━ 45s 12ms/step - accuracy: 0.7234 - loss: 0.8765
...
Epoch 10/10
3525/3525 ━━━━━━━━━━━━━━━━━━━━ 43s 12ms/step - accuracy: 0.9512 - loss: 0.1543

Test accuracy: 92.34%
Model saved successfully!
```

---

### 📸 Step 2: Prepare Your Handwriting Samples

**Requirements:**
- Write each letter (A-Z) on white paper
- Use black/dark pen or marker
- Write clearly, one letter per image
- Take photos in good lighting
- Minimum 50 samples per letter (more is better)

**File organization:**
```
handwriting_samples/
├── A/
│   ├── a_001.jpg
│   ├── a_002.jpg
│   └── ...
├── B/
│   ├── b_001.jpg
│   └── ...
└── Z/
    └── ...
```

**Image requirements:**
- Format: JPG, PNG, or JPEG
- Resolution: At least 100×100 pixels
- Background: White or light-colored
- Single character per image

---

### 🎯 Step 3: Fine-Tune with Your Handwriting

Update the folder path in `test.py`:
```python
# Line ~15 in test.py
custom_data_path = './handwriting_samples'  # Update this path
```

Then run:
```bash
python test.py
```

**What it does:**
1. Loads your handwriting samples
2. Preprocesses images (grayscale, threshold, resize)
3. Fine-tunes the base model
4. Saves personalized model
5. Generates fine-tuning visualizations

**Generated Files:**
- ✅ `handwriting_model_personalized.keras` - Your personalized model
- ✅ `finetuning_history.png` - Transfer learning progress

**Expected Output:**
```
Found 1300 images across 26 classes
Preprocessing images...
100%|████████████████████| 1300/1300 [00:15<00:00, 86.67it/s]

Starting fine-tuning...
Epoch 1/5
65/65 ━━━━━━━━━━━━━━━━━━━━ 8s 115ms/step - accuracy: 0.8945 - loss: 0.3421
...
Epoch 5/5
65/65 ━━━━━━━━━━━━━━━━━━━━ 7s 110ms/step - accuracy: 0.9734 - loss: 0.0876

Personalized model saved!
```

---

### 🔍 Step 4: Test Handwriting Recognition

Update the image path in `test_handwriting.py`:
```python
# Line ~10 in test_handwriting.py
image_path = './test_images/sample_letter.jpg'  # Your test image
```

Then run:
```bash
python test_handwriting.py
```

**What it does:**
1. Loads the personalized model
2. Preprocesses your test image
3. Predicts the letter
4. Shows top 5 predictions with confidence
5. Displays preprocessing visualization

**Output Example:**
```
Loading model...
Processing image: ./test_images/sample_letter.jpg

✅ PREDICTION RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predicted Letter: A

Top 5 Predictions:
  1. A - 98.45%
  2. H - 0.89%
  3. R - 0.34%
  4. N - 0.18%
  5. M - 0.08%

✅ Preprocessing visualization saved as 'preprocessing_steps.png'
```

**Visual Output:**

The script generates `preprocessing_steps.png` showing:
```
[Original] → [Grayscale] → [Inverted] → [Threshold] → [Cropped] → [Final 28×28]
```

---

## 📂 Project Structure
```
handwriting-ai/
│
├── 📂 handwriting_samples/          # Your training data
│   ├── A/
│   │   ├── a_001.jpg
│   │   └── ...
│   ├── B/
│   └── Z/
│
├── 📂 test_images/                  # Test images for prediction
│   └── sample_letter.jpg
│
├── 📂 models/                       # Saved models (auto-created)
│   ├── handwriting_emnist_model_improved.keras
│   └── handwriting_model_personalized.keras
│
├── 📂 visualizations/               # Generated plots (auto-created)
│   ├── training_history.png
│   ├── sample_predictions.png
│   ├── finetuning_history.png
│   └── preprocessing_steps.png
│
├── 📜 test_cnn_fashionmnist.py     # Base model training
├── 📜 test.py                       # Fine-tuning script
├── 📜 test_handwriting.py           # Prediction script
├── 📜 requirements.txt              # Dependencies
├── 📜 README.md                     # This file
├── 📜 LICENSE                       # MIT License
└── 📜 .gitignore                    # Git ignore rules
```

---

## 🐛 Troubleshooting

### ❌ Installation Issues

<details>
<summary><b>Problem: "pip not recognized"</b></summary>

**Solution:**
1. Reinstall Python with "Add to PATH" checked
2. OR use: `python -m pip install -r requirements.txt`

</details>

<details>
<summary><b>Problem: TensorFlow installation fails</b></summary>

**Solution:**
- Check Python version: `python --version` (must be 3.9-3.11)
- Update pip: `python -m pip install --upgrade pip`
- Install specific version: `pip install tensorflow==2.13.0`

**For Apple Silicon (M1/M2):**
```bash
pip install tensorflow-macos tensorflow-metal
```

</details>

<details>
<summary><b>Problem: OpenCV import error</b></summary>

**Solution:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

</details>

---

### ❌ Training Issues

<details>
<summary><b>Problem: "Out of Memory" during training</b></summary>

**Solution:**
Reduce batch size in training script:
```python
# In test_cnn_fashionmnist.py or test.py
batch_size = 16  # Change from 32 to 16
```

</details>

<details>
<summary><b>Problem: Low accuracy on personalized model</b></summary>

**Solution:**
1. **Add more samples** - Need at least 100 per letter
2. **Improve image quality** - Good lighting, clear writing
3. **Consistent style** - Write all letters similarly
4. **Remove noise** - Clean white background

</details>

<details>
<summary><b>Problem: Model file not found</b></summary>

**Solution:**
Make sure you ran training first:
```bash
python test_cnn_fashionmnist.py  # Run this first
```

Check if model exists:
```bash
ls *.keras  # macOS/Linux
dir *.keras  # Windows
```

</details>

---

### ❌ Prediction Issues

<details>
<summary><b>Problem: Wrong predictions</b></summary>

**Solution:**
1. Check preprocessing visualization (`preprocessing_steps.png`)
2. Ensure letter is centered and clear
3. Try with different lighting/contrast
4. Retrain personalized model with more samples

</details>

<details>
<summary><b>Problem: "Image not found" error</b></summary>

**Solution:**
Verify image path is correct:
```python
import os
print(os.path.exists('test_images/sample_letter.jpg'))  # Should print True
```

</details>

---

## 📌 Current Limitations

### ⚠️ What's NOT Implemented Yet

| Feature | Status | Planned Phase |
|---------|--------|---------------|
| Android application | ❌ Not started | Phase 2 |
| Handwriting generation (GAN) | ❌ Not started | Phase 2 |
| Similarity verification | ❌ Not started | Phase 3 |
| Multi-language support | ❌ Not started | Phase 3 |
| Cloud integration | ❌ Not started | Phase 3 |
| Real-time camera recognition | ❌ Not started | Phase 2 |

### ⚠️ Known Issues

- **Limited letters:** Works only for A-Z uppercase (some letters excluded: C, D, F, G, J, P, R, S, Z due to similarity issues)
- **Single character only:** Cannot process words or sentences
- **Requires preprocessing:** Raw photos need good contrast and lighting
- **No lowercase support:** Only uppercase letters recognized
- **Desktop only:** No mobile integration yet

---

## 🎯 Future Scope & Roadmap

### Phase 2 - Android Integration (Q2 2026)

- [ ] Android Studio project setup
- [ ] TensorFlow Lite model conversion
- [ ] Camera integration with CameraX
- [ ] Real-time character recognition
- [ ] Local SQLite database
- [ ] User profile management

### Phase 3 - Generation & Verification (Q3 2026)

- [ ] GAN-based handwriting generation
  - Conditional GAN (cGAN) for letter control
  - Style transfer from user samples
- [ ] Siamese Network for verification
  - Similarity percentage calculation
  - Handwriting authentication system
- [ ] Batch processing capability

### Phase 4 - Advanced Features (Q4 2026)

- [ ] Multi-language support (Devanagari, Arabic, Chinese)
- [ ] Word and sentence recognition
- [ ] Cloud-based model training
- [ ] API integration for third-party apps
- [ ] Handwriting style analysis
- [ ] Forgery detection system

---

## 👨‍💻 Project Team

<table>
<tr>
<td align="center">
<a href="https://www.linkedin.com/in/sayandip-saha-523ab430b/">
<img src="https://github.com/sayandipsaha.png" width="100px;" alt="Sayandip Saha"/><br />
<sub><b>Sayandip Saha</b></sub>
</a><br />
<sub>4th Year IT</sub><br/>
<sub>St. Thomas' College of Engineering and Technology</sub>
</td>
<td align="center">
<a href="https://www.linkedin.com/in/agniva-acherjee-2570b233b/">
<img src="https://github.com/agnivaacherjee.png" width="100px;" alt="Agniva Acherjee"/><br />
<sub><b>Agniva Acherjee</b></sub>
</a><br />
<sub>4th Year IT</sub><br/>
<sub>St. Thomas' College of Engineering and Technology</sub>
</td>
<td align="center">
<a href="https://www.linkedin.com/in/md-farhann-akhter-81b5ab303/">
<img src="https://github.com/mdfarhannakhter.png" width="100px;" alt="Md Farhann Akhter"/><br />
<sub><b>Md Farhann Akhter</b></sub>
</a><br />
<sub>4th Year IT</sub><br/>
<sub>St. Thomas' College of Engineering and Technology</sub>
</td>
</tr>
</table>

### 🙏 Acknowledgments

- **TensorFlow Team** - Deep learning framework
- **EMNIST Dataset** - Training data
- **OpenCV Community** - Image processing tools
- **St. Thomas' College** - Academic support and guidance

---

## 📊 Model Performance

### Base Model (EMNIST)

| Metric | Value |
|--------|-------|
| Training Accuracy | ~95% |
| Validation Accuracy | ~92% |
| Test Accuracy | ~92-94% |
| Training Time (CPU) | ~8-10 minutes |
| Training Time (GPU) | ~2-3 minutes |
| Model Size | ~2.5 MB |

### Personalized Model

| Metric | Value |
|--------|-------|
| Accuracy (on user data) | ~95-98% |
| Fine-tuning Time | ~3-5 minutes |
| Required Samples | Minimum 50/letter |
| Model Size | ~2.6 MB |

---

## 📞 Contact & Support

### Connect With the Team

[![LinkedIn - Sayandip](https://img.shields.io/badge/LinkedIn-Sayandip%20Saha-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/sayandip-saha-523ab430b/)
[![LinkedIn - Agniva](https://img.shields.io/badge/LinkedIn-Agniva%20Acherjee-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/agniva-acherjee-2570b233b/)
[![LinkedIn - Farhann](https://img.shields.io/badge/LinkedIn-Md%20Farhann%20Akhter-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/md-farhann-akhter-81b5ab303/)
[![GitHub](https://img.shields.io/badge/GitHub-Handwriting%20AI-100000?style=for-the-badge&logo=github)](https://github.com/yourusername/handwriting-ai)

### Found a Bug?

[Report Issue](https://github.com/yourusername/handwriting-ai/issues/new) • [Request Feature](https://github.com/yourusername/handwriting-ai/issues/new?labels=enhancement)

---

## 🔖 Keywords

`handwriting-recognition` `deep-learning` `cnn` `tensorflow` `opencv` `ocr` `transfer-learning` `emnist` `image-processing` `python` `machine-learning` `computer-vision` `ai` `android-app` `final-year-project`

---

## 📚 References & Resources

### Papers & Research
- [EMNIST Dataset Paper](https://arxiv.org/abs/1702.05373)
- [Transfer Learning in Deep Learning](https://arxiv.org/abs/1411.1792)
- [Handwriting Recognition Survey](https://arxiv.org/abs/2012.13880)

### Tutorials
- [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)
- [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

<div align="center">

### ⭐ If you found this project helpful, please consider starring the repository!

**Made with ❤️ by Sayandip Saha, Agniva Acherjee & Md Farhann Akhter**

*Final Year Project | St. Thomas' College of Engineering and Technology*

*Last Updated: February 2026 | Version 1.0.0 (Phase 1)*

[![GitHub stars](https://img.shields.io/github/stars/yourusername/handwriting-ai?style=social)](https://github.com/yourusername/handwriting-ai/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/handwriting-ai?style=social)](https://github.com/yourusername/handwriting-ai/network/members)

</div>
