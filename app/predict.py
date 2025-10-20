import joblib
import pandas as pd

# Load trained model
def load_model(path="model/model.pkl"):
    return joblib.load(path)

# Make a prediction on a new flight
def predict_delay(model, flight_data):
    """
    flight_data: dict
      example: {"ORIGIN": "JFK", "DEST": "LAX", "CARRIER": "DL", "DAY_OF_WEEK": 3, "DEP_HOUR": 15}
    """
    df = pd.DataFrame([flight_data])
    df_encoded = pd.get_dummies(df)

    # Ensure same columns as training
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_columns]

    prediction = model.predict(df_encoded)[0]
    probability = model.predict_proba(df_encoded)[0][1]  # probability of delay
    return prediction, probability
