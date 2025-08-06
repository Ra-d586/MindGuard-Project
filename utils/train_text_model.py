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

# Dummy dataset for text + emotion
# Replace with actual real dataset if available
data = {
    "text": [
        "I feel great today", "I'm so tired and down", "Not sure what I feel", 
        "Life is amazing", "I’m anxious about tomorrow", "I’m okay, I guess",
        "I feel very low", "I’m excited about the weekend", "Nothing makes sense",
        "I am really relaxed and calm"
    ],
    "label": ["happy", "sad", "neutral", "happy", "stress", "neutral", "sad", "happy", "sad", "happy"]
}
df = pd.DataFrame(data)

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
