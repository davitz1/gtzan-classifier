
# Music Genre Classifier

CNN-based music genre classification using MFCC features on the GTZAN dataset.

## Table of Contents
- [Setup](#setup)
- [Dataset Download](#dataset-download)
- [Feature Extraction](#feature-extraction)
- [Training](#training)
- [Inference](#inference)
- [Web Application](#web-application)
- [How It Works](#how-it-works)

---

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/gtzan-classifier.git
cd gtzan-classifier
```
### 2. Create Virtual Environment (Recommended)
```python
# Using conda
conda create -n gtzan python=3.10
conda activate gtzan

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
```python
pip install -r requirements.txt
```
---
## Dataset Download
GTZAN can be download through Kaggle API or manually.

### Manual Download
1. Visit https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
2. Download and extract the zip file
3. Move the `genres_original` folder to your project so it looks like `gtzan-classifier/data/gtzan/genres_original/`

### API Download
**API Option Requirements:**
1. Kaggle account 
2. API credentials

**SETUP Kaggle API:**
1. Go to https://www.kaggle.com/settings/account
2. Click 'Create New API Token'
3. Create a `kaggle.json` file manually: 
**Windows**:
```bash
mkdir %USERPROFILE%\.kaggle
notepad %USERPROFILE%\.kaggle\kaggle.json
```
**Linux/Mac**:
```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
```
4. Paste this content (replace with your credentials):
```json
{ 
 "username": "your_kaggle_username",
    "key": "your_api_key_here"
}
```
5. Set correct permissions (Linux/Mac only):
```bash
chmod 600 ~/.kaggle/kaggle.json
```
**Download the dataset:**
```bash
#install kaggle API
pip install kaggle

#run the download script
python download_dataset.py
```
This will:
- Download 1.2 GB of audio files
- Extract to `genres_original` 
- Organize into 10 genre folders
- Verify the structure
---
## Feature Extraction
Extract MFCC features from the audio files:
```bash
python src/data_processing/feature_extractor.py
```
This will:
- Load each 30-second audio file
- Split into 5 segments of 6 seconds each
- Extract 13 MFCC coefficients per segment
- Saves to `mfcc_data.npz`
---
## Training
Train the CNN model:
```bash
python src/training/train_mfcc.py
```
Training configuration:
- Batch size: 32
- Epochs: 30
- Learning rate: 0.001
- Train/Val/Test split: 60%/20%/20%
- Device: CUDA if available, else CPU

Output:
- Saved model: `mfcc_cnn_trained.pth`
- Plots: `plots`

**Expected accuracy:** ~77% on test set

---
## Inference
Classify a single audio file:
```bash
python src/inference/infer_mfcc.py path/to/your/song.wav
```
Python Script
```python
from src.inference.infer_mfcc import GenreClassifier

# Load model
classifier = GenreClassifier("outputs/mfcc_cnn/mfcc_cnn_trained.pth")

# Classify a song
result = classifier.predict("song.wav", return_probabilities=True)

print(f"Genre: {result['genre']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```
---
## Web Application
Start the Server
```bash
uvicorn src.app.main:app --reload
```
Access: Open http://localhost:8000 in your browser

Features:
- Upload audio file: Drag & drop or click to select
- Supported formats: .wav (recommended), .mp3, .flac, .ogg, .m4a
- Real-time prediction: Results appear instantly
---
## How It Works
### 1. **Feature Extraction (MFCC)**

MFCC (Mel-Frequency Cepstral Coefficients) captures the timbral characteristics of audio.
-Input: 30-second audio clip
-Process: Split into 5 segments (6s each) -> Extract 13 MFCCs
-Output: (5, 130, 13) matrix

### 2. **Model Architecture (CNN)**

A Convolutional Neural Network processes the MFCC image-like data:

 - Three Convolutional Layers (extract features)

- Flatten Layer
- 2 Fully Connected Layers (classification)
- Softmax Output (10 genres)

### 3. **Inference Logic**

The model was trained on 30-second audio clips. When you upload a file:

**For Short Audio (< 30 seconds):**

Padded with silence to reach 30 seconds.

**For Long Audio (> 30 seconds):**

Truncated to the first 30 seconds.
Only the intro/beginning is analyzed.
The 30s clip is split into 5 segments, predictions are averaged.

**Why?**

The model was trained only on 30-second clips from GTZAN.
Most songs establish their genre in the first 30 seconds (intro, instrumentation, rhythm).

---

## Citation


Tzanetakis, G., & Cook, P. (2002). **Musical genre classification of audio signals**. *IEEE Transactions on Speech and Audio Processing*, *10*(5), 293-302.

--- 
## License

This project is for educational purposes. 

**Dataset:** GTZAN has its own terms of use for research purposes.

**Code:** MIT License