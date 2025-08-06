import joblib
import pandas as pd

model = joblib.load("models/behavior_model.pkl")

def predict_behavior_emotion(behavior_dict):
    df = pd.DataFrame([behavior_dict])
    prediction = model.predict(df)[0]
    return prediction
