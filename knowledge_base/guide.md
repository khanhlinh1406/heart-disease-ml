# ❤️ Heart Disease ML Project – Learning & Review Guide

## 🎯 1. Project Overview

This is a beginner-friendly Machine Learning project focused on predicting whether a patient has heart disease based on medical features.

### 🔍 Objective

* Build a binary classification model (0: No Disease, 1: Disease)
* Understand end-to-end ML pipeline
* Learn how to evaluate and improve a model

---

## 🧠 2. Core ML Concepts You Must Understand

### 2.1 Supervised Learning

* Input: Features (X)
* Output: Label (y)
* Goal: Learn mapping from X → y

---

### 2.2 Classification Problem

* Binary classification: 0 or 1
* Example: Disease vs No Disease

---

### 2.3 Logistic Regression

#### 📌 Idea

Model predicts probability:

P(y=1) = 1 / (1 + e^(-z))

Where:

* z = w1x1 + w2x2 + ... + b

#### 📌 Why use it?

* Simple
* Interpretable
* Good baseline

---

### 2.4 Feature Scaling (Standardization)

#### 📌 Formula

x_scaled = (x - mean) / std

#### 📌 Why?

* Helps model converge faster
* Prevents feature dominance

---

### 2.5 Train/Test Split

* Train: 80%
* Test: 20%

#### ⚠️ Important Rule

Never use test data during training

---

### 2.6 Data Leakage

#### ❌ Wrong

Fit scaler before splitting

#### ✅ Correct

Split → Fit scaler on train → Transform test

---

## 🔄 3. Project Pipeline

```
preprocess.py → train.py → evaluate.py → predict.py
```

---

## ⚙️ 4. Step-by-Step Implementation

### 4.1 Preprocessing (`preprocess.py`)

#### Tasks:

* Load dataset
* Split features & target
* Train/test split
* Scale features
* Save scaler

#### Key Code Logic:

```python
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(...)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 4.2 Training (`train.py`)

#### Tasks:

* Load processed data
* Train model
* Save model

#### Key Code Logic:

```python
model = LogisticRegression()
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
```

---

### 4.3 Evaluation (`evaluate.py`)

#### Metrics:

##### 1. Accuracy

* Overall correctness

##### 2. Confusion Matrix

|          | Pred 0 | Pred 1 |
| -------- | ------ | ------ |
| Actual 0 | TN     | FP     |
| Actual 1 | FN     | TP     |

##### 3. Precision

TP / (TP + FP)

##### 4. Recall

TP / (TP + FN)

##### 5. F1-score

2 * (Precision * Recall) / (Precision + Recall)

---

### 4.4 Prediction (`predict.py`)

#### Tasks:

* Load model & scaler
* Transform input
* Predict probability
* Apply threshold

---

## 🔥 5. Threshold Tuning (IMPORTANT)

### Default:

```
threshold = 0.5
```

### Custom:

```
y_pred = (prob > 0.4)
```

### Trade-off:

| Threshold | Recall | Precision |
| --------- | ------ | --------- |
| Lower     | ↑      | ↓         |
| Higher    | ↓      | ↑         |

### In medical context:

* Prioritize Recall (detect disease)

---

## 📊 6. Your Model Result Analysis

* Accuracy: ~85%
* Recall (Disease): High (~0.89)

### Interpretation:

* Model detects most disease cases
* Some false positives exist (acceptable)

---

## 🚀 7. Improvements Ideas

### 7.1 Model Improvements

* Random Forest
* XGBoost

### 7.2 Feature Engineering

* Remove noise
* Add domain knowledge

### 7.3 Threshold Optimization

* Try 0.3 → 0.6
* Choose based on recall

---

## 🧪 8. Advanced Topics to Learn Next

* ROC Curve & AUC
* Precision-Recall Curve
* Cross-validation
* Hyperparameter tuning (GridSearchCV)

---

## 🏗️ 9. Production Mindset

### Separate phases:

* Training
* Evaluation
* Inference

### Artifacts:

* model.pkl
* scaler.pkl

---

## 🧠 10. Key Takeaways

* Always avoid data leakage
* Use probability instead of raw prediction
* Tune threshold based on problem
* Evaluation matters more than training

---

## 💡 11. Suggested Next Steps

* Build API with FastAPI
* Create UI (Angular)
* Deploy model
* Add logging & monitoring

---

## 🎯 Final Thought

This project demonstrates a complete ML pipeline and builds a strong foundation for real-world machine learning systems.
