# utils/predict_text.py

import torch
import joblib
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

# Load tokenizer and BERT model
tokenizer = DistilBertTokenizer.from_pretrained("models")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Load logistic regression model and label encoder
model = joblib.load("models/text_model.pkl")
label_encoder = joblib.load("models/text_label_encoder.pkl")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return emb

def predict_text_emotion(text):
    emb = get_bert_embedding(text).reshape(1, -1)
    pred = model.predict(emb)[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label
