from preprocess import preprocess_data
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def evaluate():
    THRESHOLD = 0.4
    X_train, X_test, y_train, y_test = preprocess_data()
    model = joblib.load("model.pkl")
    
    prob = model.predict_proba(X_test)[:, 1]
    y_pred = (prob > THRESHOLD).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\n\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    evaluate()