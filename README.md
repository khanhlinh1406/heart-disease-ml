# Heart Disease ML

A beginner-friendly Machine Learning project to predict heart disease using Logistic Regression. This repo demonstrates a clean end-to-end ML pipeline: preprocessing → training → evaluation → prediction.

---

## 📦 1. Project Structure

```
heart-disease-ml/
│
├── data/
│   └── heart.csv            # Dataset
│
├── preprocess.py            # Data preprocessing (split, scale, save scaler)
├── train.py                 # Train model and save model.pkl
├── evaluate.py              # Evaluate model (accuracy, confusion matrix, report)
├── predict.py               # Predict on new sample (with threshold)
│
├── model.pkl                # Saved model (generated after training)
├── scaler.pkl               # Saved scaler (generated after preprocessing)
│
├── requirements.txt         # Dependencies
└── README.md
```

---

## ⚙️ 2. Setup & Installation

### 2.1 Clone repo

```bash
git clone https://github.com/khanhlinh1406/heart-disease-ml.git
cd heart-disease-ml
```

### 2.2 Create virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2.3 Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:

```bash
pip install pandas scikit-learn numpy joblib
```

---

## 🚀 3. How to Run

### Step 1: Preprocess data

```bash
python preprocess.py
```

* Split train/test (80/20)
* Scale features using `StandardScaler`
* Save `scaler.pkl`

---

### Step 2: Train model

```bash
python train.py
```

* Train Logistic Regression model
* Save `model.pkl`

---

### Step 3: Evaluate model

```bash
python evaluate.py
```

* Print:

  * Accuracy
  * Confusion Matrix
  * Classification Report

* Supports custom threshold (edit in file or pass param)

---

### Step 4: Predict new data

```bash
python predict.py
```

* Load `model.pkl` and `scaler.pkl`
* Predict probability and class
* Apply custom threshold (default: 0.4)

---

## 🧪 4. Test with Custom Data

### 4.1 Sample format

```python
import numpy as np

sample = np.array([[
    52,  # age
    1,   # sex
    0,   # cp (chest pain type)
    125, # trestbps
    212, # chol
    0,   # fbs
    1,   # restecg
    168, # thalach
    0,   # exang
    1.0, # oldpeak
    2,   # slope
    2,   # ca
    3    # thal
]])
```

> ⚠️ Important: Feature order must match training data exactly.

---

### 4.2 Run prediction manually

```python
from predict import predict

pred, prob = predict(sample, threshold=0.4)

print("Prediction:", "Disease" if pred == 1 else "No Disease")
print(f"Probability: {prob:.2f}")
```

---

## 🎯 5. Threshold Tuning (Optional)

Default threshold = `0.5`

You can change to improve recall (important for medical cases):

```python
pred, prob = predict(sample, threshold=0.4)
```

### Trade-off:

| Threshold        | Recall                | Precision           |
| ---------------- | --------------------- | ------------------- |
| Lower (0.3–0.4)  | ↑ detect more disease | ↓ more false alarms |
| Higher (0.6–0.7) | ↓ miss more disease   | ↑ more precise      |

---

## 📊 6. Model Output Example

```
Accuracy: ~0.85
Confusion Matrix:
[[81 19]
 [12 93]]
```

* Good recall for disease detection
* Some false positives (acceptable in medical context)

---

## 🧠 7. Key Concepts Covered

* Supervised Learning (Classification)
* Logistic Regression
* Feature Scaling (StandardScaler)
* Train/Test Split
* Data Leakage (and how to avoid it)
* Evaluation Metrics (Precision, Recall, F1)
* Threshold Tuning

---

## 📌 8. Notes

* Always run `preprocess.py` before `train.py`
* Ensure `model.pkl` and `scaler.pkl` exist before prediction
* Do not fit scaler on full dataset (avoid data leakage)

---

