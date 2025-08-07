# utils/train_text_model.py

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import os
from tqdm import tqdm
from datasets import load_dataset



dataset = load_dataset("go_emotions", "raw")



# Convert to DataFrame
df = dataset['train'].to_pandas()
print("Available columns:", df.columns.tolist())

# List of all possible emotions (columns with 0/1 values)
emotion_cols = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

# Keep only rows with a single label (sum == 1)
df['label_count'] = df[emotion_cols].sum(axis=1)
df = df[df['label_count'] == 1]

# Get the label name (emotion with value == 1)
df['emotion'] = df[emotion_cols].idxmax(axis=1)

# Filter to core emotions
# Filter to core emotions
core_emotions = ['joy', 'sadness', 'anger', 'neutral', 'fear']
df = df[df['emotion'].isin(core_emotions)]

# Rename for consistency
df = df.rename(columns={"text": "text", "emotion": "label"})

# Reduce dataset size to speed up prototyping
df = df.sample(n=2000, random_state=42).reset_index(drop=True)

print(f"✅ Loaded {len(df)} labeled examples from GoEmotions (core emotions only)")





# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["label"])

# Tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Function to extract embeddings
def get_bert_embeddings(texts):
    embeddings = []
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = bert(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(emb)
    return embeddings

X = get_bert_embeddings(df["text"])

# Train simple classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/text_model.pkl")
joblib.dump(le, "models/text_label_encoder.pkl")
tokenizer.save_pretrained("models/")

print("✅ Text model and tokenizer saved to models/")
