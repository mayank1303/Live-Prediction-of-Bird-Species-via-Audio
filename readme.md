# Live Prediction of Bird Species via Audio

A simple Flask-based demo that classifies **live audio** (or uploaded clips) into bird species. The app turns audio into features (e.g., spectrogram/MFCC) and feeds them to a trained model to predict the most likely species — right from your browser.

> ⚠️ Note: This is a student/demo project. Accuracy depends on the dataset and model; expect misclassifications on noisy or unseen sounds.

---

## ✨ Features
- Web UI (Flask) for quick testing in the browser
- Live or file-based audio inference flow
- Preprocessing pipeline for audio (feature extraction)
- Easily swappable model checkpoint

---

## 📦 Tech Stack
- **Python**
- **Flask** for the web app
- **Audio/ML tooling**: `librosa`, `numpy`, `scikit-learn` (and/or `tensorflow/keras` if your model uses it)

> Exact libraries depend on your model code. See `audio_flask_classification.py` and `app.py` for imports once you finalize dependencies.

---

## 📂 Project Structure
"""
├─ app.py # Flask app (routes/views)
├─ audio_flask_classification.py # Audio preprocessing + inference helpers
├─ main.py # Entry point to run the server
└─ README.md
"""




 

## Group Members
1. [Megha Sharma](https://github.com/m36h4)
2. [Mayank Kumar Rathor](https://github.com/mayank1303)
3. Saloni Sisodiya
