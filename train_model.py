# train_model.py

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X.squeeze() + 1 + 0.1 * np.random.randn(100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Log parameters and metrics using MLflow
with mlflow.start_run():
    mlflow.log_param("alpha", model.intercept_)
    mlflow.log_param("beta", model.coef_[0])
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
