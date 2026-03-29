# import pandas for data loading and manipulation
import pandas as pd
# import StandardScaler to normalize (scale) feature values
from sklearn.preprocessing import StandardScaler
# import function to split dataset into training and testing sets
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data():
    # load dataset from CSV file
    df = pd.read_csv("data/heart.csv")

    # separate input features (X) by dropping the target column
    x = df.drop("target", axis=1)
    # extract the target variable (labels)
    y = df["target"]


    # split the dataset into training and testing sets
    # - 80% for training, 20% for testing
    # - random_state ensures reproducibility
    # - stratify=y keeps the same class distribution in both sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # initialize the scaler (used to standardize features)
    scaler = StandardScaler()

    # scale the features so that each has mean = 0 and standard deviation = 1
    # this helps the model learn more efficiently
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test) 
    
    # to save scaler
    joblib.dump(scaler, "scaler.pkl")

    return x_train_scaled, x_test_scaled, y_train, y_test


# just only run preprocess_data() when runnning this file directly
if __name__ == "__main__":
    preprocess_data()