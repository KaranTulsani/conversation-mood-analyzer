from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import torch.nn.functional as F
import re
from fastapi.middleware.cors import CORSMiddleware


from sentence_transformers import SentenceTransformer
from src.lstm_model import EmotionLSTM
from src.sentiment_map import SENTIMENT_LABELS

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="Conversation Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://conversation-sentiment-analyzer.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Text normalization
# -------------------------------
def normalize_words(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return set(text.split())

NEGATIVE_WORDS = {
    "tired", "stress", "stressful", "sad", "angry", "upset",
    "exhausted", "bad", "terrible", "worried"
}

POSITIVE_WORDS = {
    "hope", "hoping", "improve", "happy", "good", "great",
    "better", "excited", "love"
}

# -------------------------------
# Load model & embedder ONCE
# -------------------------------
model = EmotionLSTM(
    input_dim=384,
    hidden_dim=128,
    num_classes=3
)
model.load_state_dict(torch.load("data/processed/emotion_lstm.pt"))
model.eval()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Request / Response schema
# -------------------------------
class ConversationRequest(BaseModel):
    conversation: List[str]

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/predict")
def analyze_sentiment(request: ConversationRequest):
    sentences = request.conversation

    embeddings = embedder.encode(sentences)
    X = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=-1).squeeze(0)

    results = []

    for i, sentence in enumerate(sentences):
        pred = torch.argmax(probs[i]).item()
        words = normalize_words(sentence)

        # Override neutral if strong lexical cue
        if pred == 0:
            if words & NEGATIVE_WORDS:
                pred = 1
            elif words & POSITIVE_WORDS:
                pred = 2

        results.append({
            "text": sentence,
            "sentiment": SENTIMENT_LABELS[pred]
        })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
