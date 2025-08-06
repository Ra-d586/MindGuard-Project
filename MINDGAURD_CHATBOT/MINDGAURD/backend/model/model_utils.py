import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

# Initialize variables with None to handle loading failures
text_model = None
label_encoder = None
vectorizer = None

# Try to load the text model and related files
try:
    with open('model/text_model.pkl', 'rb') as f:
        text_model = pickle.load(f)
    with open('model/text_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('model/tokenizer_config.json', 'r') as f:
        tokenizer_config = json.load(f)
    with open('model/vocab', 'r') as f:
        vocab = f.read().splitlines()

    # Custom TF-IDF vectorizer based on vocab
    vectorizer = TfidfVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
except (pickle.UnpicklingError, FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Warning: Failed to load model or related files: {e}. Using dummy logic as fallback.")
    # Fallback: Define dummy label encoder and vectorizer
    label_encoder = {'sad': 0, 'neutral': 1, 'happy': 2}
    vocab = ['sad', 'neutral', 'happy', 'feel', 'today']
    vectorizer = TfidfVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())

def predict_mood_from_text(text):
    global text_model, label_encoder, vectorizer
    if text_model is None or label_encoder is None or vectorizer is None:
        # Fallback dummy prediction
        if 'sad' in text.lower():
            return 'sad', "Hey, you seem a bit down. Why are you sad? What happened today?"
        elif 'happy' in text.lower():
            return 'happy', "Great to see you're doing well! How can I make your day even better?"
        else:
            return 'neutral', "Hi! You seem okay today. How can I assist you?"
    
    # Vectorize the input text
    text_vector = vectorizer.transform([text])
    # Predict mood using the text model
    mood_idx = text_model.predict(text_vector)[0]
    mood = list(label_encoder.keys())[list(label_encoder.values()).index(mood_idx)] if mood_idx in label_encoder.values() else 'neutral'
    messages = {
        'sad': "Hey, you seem a bit down. Why are you sad? What happened today?",
        'neutral': "Hi! You seem okay today. How can I assist you?",
        'happy': "Great to see you're doing well! How can I make your day even better?"
    }
    return mood, messages.get(mood, "Hi! Let's chat about your day!")

# Dummy mood analysis from smartwatch data
def analyze_mood(data):
    hrv = data.get('HRV', 60)
    steps = data.get('steps', 5000)
    sleep_hours = data.get('sleep_hours', 6.5)
    screen_time = data.get('screen_time', 200)

    if hrv < 50 or steps < 2000 or sleep_hours < 5 or screen_time > 300:
        return "sad", "Hey, you seem a bit down based on your data. Why are you sad?"
    elif 50 <= hrv <= 70 and 2000 <= steps <= 8000 and 5 <= sleep_hours <= 7 and screen_time <= 200:
        return "neutral", "Hi! Your data suggests you're okay today. How can I assist?"
    else:
        return "happy", "Great to see your data indicates you're doing well!"
