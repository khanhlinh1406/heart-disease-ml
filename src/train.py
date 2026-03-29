# import logistic regression model (used for classification problems)
from sklearn.linear_model import LogisticRegression
from preprocess import preprocess_data
import joblib

x_train, x_test, y_train, y_test = preprocess_data()

# initialize Logistic Regression model
# max_iter=1000 increases the number of training iterations to ensure convergence
model = LogisticRegression(max_iter=1000)


# train the model using training data
# the model learns the relationship between input features (X) and target (y)
model.fit(x_train, y_train)

# to save trained model
joblib.dump(model, "model.pkl")
