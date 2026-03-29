# import function to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# import logistic regression model (used for classification problems)
from sklearn.linear_model import LogisticRegression

# import StandardScaler to normalize (scale) feature values
from sklearn.preprocessing import StandardScaler

# import pandas for data loading and manipulation
import pandas as pd


# load dataset from CSV file
df = pd.read_csv("data/heart.csv")


# separate input features (X) by dropping the target column
x = df.drop("target", axis=1)


# initialize the scaler (used to standardize features)
scaler = StandardScaler()


# scale the features so that each has mean = 0 and standard deviation = 1
# this helps the model learn more efficiently
x_scaled = scaler.fit_transform(x)


# extract the target variable (labels)
y = df["target"]


# split the dataset into training and testing sets
# - 80% for training, 20% for testing
# - random_state ensures reproducibility
# - stratify=y keeps the same class distribution in both sets
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# initialize Logistic Regression model
# max_iter=1000 increases the number of training iterations to ensure convergence
model = LogisticRegression(max_iter=1000)


# train the model using training data
# the model learns the relationship between input features (X) and target (y)
model.fit(x_train, y_train)

# to save trained model and scaler
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
