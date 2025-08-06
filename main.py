from utils.predict_text import predict_text_emotion
from utils.predict_behavior import predict_behavior_emotion
from utils.fusion_predict import fuse_predictions
from utils.chatbot import generate_response
from utils.logger import log_feedback

import pandas as pd

# Example inputs
user_text = input("You: ")
behavior_data = {
    'HRV': 65,
    'steps': 6000,
    'screen_time': 180,
    'sleep_hours': 6,
    'time_at_home_hrs': 16,
    'time_outside_hrs': 2,
    'calls_count': 5,
    'call_duration': 15,
    'messages_count': 20
}

# Step 1: Predict emotions
text_emotion = predict_text_emotion(user_text)
behavior_emotion = predict_behavior_emotion(behavior_data)
final_mood = fuse_predictions(text_emotion, behavior_emotion)

# Step 2: Generate chatbot response
response = generate_response(user_text, final_mood)
print("MindPulse:", response)

# Step 3: Confirm feedback
confirm = input("Are you actually feeling this way? (yes/no): ").strip().lower()
if confirm == 'yes':
    log_feedback(behavior_data, final_mood)
    print("✅ Logged for training.")
else:
    print("⚠️ Not logged.")
