import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/housing.csv")

# Handle missing values
df = df.dropna()

# One-hot encode categorical column
df = pd.get_dummies(df)

# Split features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Metrics
rmse = root_mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("RMSE:", rmse)
print("R2:", r2)

# Save metrics
metrics = {
    "rmse": float(rmse),
    "r2": float(r2)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)
