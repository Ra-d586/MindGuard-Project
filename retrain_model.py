# retrain_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("data/behavioral_emotion_data.csv")

# Features and labels
features = [
    'HRV', 'steps', 'screen_time', 'sleep_hours',
    'time_at_home_hrs', 'time_outside_hrs',
    'calls_count', 'call_duration', 'messages_count'
]
X = df[features]
y = df["mood_label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model and label encoder
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/behavior_model.pkl")
joblib.dump(le, "models/behavior_label_encoder.pkl")

print("✅ Behavior model trained and saved to models/behavior_model.pkl")
