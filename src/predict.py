import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def predict(sample):
    THRESHOLD = 0.4
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    # scale input
    sample_scaled = scaler.transform(sample)

    # predict (0/1)
    prediction = model.predict(sample_scaled)

    # predict probability
    probability = model.predict_proba(sample_scaled)

    return 1 if prediction[0] >= THRESHOLD else 0, probability[0]

if __name__ == "__main__":
    sample = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])
    pred, prob = predict(sample)

    print("Prediction: ", pred)
    print("Probability: ", prob)

    prob_have_disease = prob[1]
    if prob_have_disease > 0.8:
        print("High risk")
    elif prob_have_disease > 0.6:
        print("Moderate risk")
    else:
        print("Low confidence prediction")