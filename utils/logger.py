# utils/logger.py

import os
import csv
from datetime import datetime
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrain_model import retrain_behavior_model


def log_feedback(behavior_data, mood, user_text=None):
    date = datetime.now().strftime("%Y-%m-%d")
    row = {
        'user_id': 'UX',
        'date': date,
        **behavior_data,
        'mood_label': mood
    }

    # Save to behavioral dataset
    csv_file = 'data/behavioral_emotion_data.csv'
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Save text data if provided (optional)
    if user_text:
        text_row = {'text': user_text, 'label': mood}
        text_csv = 'data/text_emotion_data.csv'
        text_exists = os.path.exists(text_csv)
        with open(text_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=text_row.keys())
            if not text_exists:
                writer.writeheader()
            writer.writerow(text_row)

    # 🔁 Retrain behavioral model automatically
    print("🔄 Retraining behavioral model...")
    retrain_behavior_model()
    print("✅ Behavioral model updated!")
