import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris()

X = pd.DataFrame(data = iris["data"], columns= iris["feature_names"])
y = pd.DataFrame(data = iris["target"], columns=["target"])

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Mlflow tracking
import sys 

with mlflow.start_run():
    
    # Specified Parameters 
    c = float(sys.argv[2]) if len(sys.argv) > 1 else 0.5 # Let the user specify C argument via Cli

    # Instanciate and fit the model 
    lr = LogisticRegression(C=c)
    lr.fit(X_train.values, y_train.values)

    # Store metrics 
    predicted_qualities = lr.predict(X_test.values)
    accuracy = lr.score(X_test.values, y_test.values)

    # Print results 
    print("LogisticRegression model")
    print("Accuracy: {}".format(accuracy))

    # Log Metric 
    mlflow.log_metric("Accuracy", accuracy)

    # Log Param
    mlflow.log_param("C", c)
