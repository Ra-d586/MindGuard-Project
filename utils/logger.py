import pandas as pd
import os

def log_feedback(behavior_dict, mood):
    filepath = "data/behavioral_emotion_data.csv"

    behavior_dict['mood_label'] = mood
    df = pd.DataFrame([behavior_dict])

    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
