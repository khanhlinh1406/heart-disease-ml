import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

sample = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])

# scale input
sample_scaled = scaler.transform(sample)

# predict (0/1)
prediction = model.predict(sample_scaled)

# predict probability
probability = model.predict_proba(sample_scaled)

print("Prediction: ", prediction[0])
print("Probability: ", probability[0])


prob_have_disease = probability[0][1]
if prob_have_disease > 0.8:
    print("High risk")
elif prob_have_disease > 0.6:
    print("Moderate risk")
else:
    print("Low confidence prediction")