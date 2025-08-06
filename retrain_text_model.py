# retrain_text_model.py

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import os
from tqdm import tqdm

def get_bert_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(emb)
    return embeddings

def retrain_text_model():
    df = pd.read_csv("data/text_emotion_data.csv")

    if df.empty or len(df) < 5:
        print("⚠️ Not enough text data to retrain.")
        return

    y = df["label"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    X = get_bert_embeddings(df["text"], tokenizer, bert)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/text_model.pkl")
    joblib.dump(le, "models/text_label_encoder.pkl")
    tokenizer.save_pretrained("models/")

    print("✅ Text model retrained and saved.")

if __name__ == "__main__":
    retrain_text_model()
