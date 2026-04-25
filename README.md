# 🤟 ISL Gesture Recognition

Real-time Indian Sign Language recognition using MediaPipe + scikit-learn + Streamlit. Detects 35 gestures (A–Z, 1–9) from a webcam and builds a sentence live.

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `collect_data.py` | Collect webcam training data |
| `2_train_model.py` | Train the ML model |
| `3_app.py` | Run the Streamlit app |
| `live_landmarks.csv` | Collected dataset (auto-generated) |
| `isl_model.pkl` | Trained model (auto-generated) |

---

## ⚙️ Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install streamlit opencv-python mediapipe scikit-learn pandas numpy matplotlib seaborn openpyxl
```

---

## 🚀 Usage — Run in Order

### 1. Collect Data
```bash
python collect_data.py
```
- 3 people, 100 samples each per gesture
- `SPACE` = start/pause | `Q` = skip gesture | `ESC` = save & quit

### 2. Train Model
```bash
python 2_train_model.py
```
- Targets 89–90% test accuracy
- Saves model, scaler, label encoder, and confusion matrix

### 3. Run App
```bash
streamlit run 3_app.py
```
- Open `http://localhost:8501` in your browser
- Press **▶ Start Camera** and begin signing

---

## 🛠 Troubleshooting

| Problem | Fix |
|---------|-----|
| No gestures detected | Lower Min Confidence slider to 0.35 |
| Feature mismatch error | Delete `.pkl` files and retrain |
| No hands detected | Improve lighting, move hand closer |
| Camera not opening | Close any other app using the webcam |

---

## 🏗 Tech Stack

MediaPipe · scikit-learn (Random Forest + MLP ensemble) · Streamlit · OpenCV · Python 3.9+

---

