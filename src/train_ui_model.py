# Importing Required Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Loading Processed Dataset
df = pd.read_csv("data/processed_cardio.csv")

# Splitting Features and Target
X = df.drop("cardio", axis=1)
y = df["cardio"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Accuracy
accuracy = model.score(X_test_scaled, y_test)

print("Deployment Model Accuracy:", round(accuracy,4))

# Save Model + Scaler
joblib.dump(model, "models/ui_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and Scaler Saved Successfully")