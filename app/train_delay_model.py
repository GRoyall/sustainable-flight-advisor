import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load small sample of BTS data (replace with your CSV path)
df = pd.read_csv("data/raw/bts_sample.csv")  

# Create a simple target: delayed if arrival delay > 15 min
df["delayed"] = (df["ARR_DELAY"] > 15).astype(int)

# Features to train on (categorical columns will be one-hot encoded)
features = ["ORIGIN", "DEST", "CARRIER", "DAY_OF_WEEK", "DEP_HOUR"]
X = pd.get_dummies(df[features])
y = df["delayed"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "model/model.pkl")
print("✅ Model saved to model/model.pkl")
