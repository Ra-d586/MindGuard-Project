# retrain_model.py

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def retrain_behavior_model():
    # Load dataset
    df = pd.read_csv("data/behavioral_emotion_data.csv")

    # Drop ID and date
    X = df.drop(columns=["user_id", "date", "mood_label"])
    y = df["mood_label"]

    # Encode mood labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_encoded)

    # Save model and encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/behavior_model.pkl")
    joblib.dump(le, "models/behavior_label_encoder.pkl")
   
    

# If run directly as a script
if __name__ == "__main__":
    retrain_behavior_model()
    print("✅ Behavior model trained and saved to models/behavior_model.pkl")
