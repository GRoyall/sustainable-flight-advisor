import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load sample or small subset of BTS data
# You can start with a CSV from BTS or your own small dataset
df = pd.read_csv("data/raw/bts_sample.csv")

# Simple example: train model to predict delays >15 minutes
df["delayed"] = (df["ARR_DELAY"] > 15).astype(int)
features = ["ORIGIN", "DEST", "CARRIER", "DAY_OF_WEEK", "DEP_HOUR"]
X = pd.get_dummies(df[features])
y = df["delayed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "model/model.pkl")
print("✅ Model saved to model/model.pkl")
