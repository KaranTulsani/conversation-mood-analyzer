from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import torch.nn.functional as F
import re
import gc
import os
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer
from src.lstm_model import EmotionLSTM
from src.sentiment_map import SENTIMENT_LABELS

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="Conversation Sentiment API")

# CORS - Must be configured before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "https://conversation-mood-analyzer.vercel.app",
        "https://conversation-mood-analyzer-git-main-karantulsanis-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
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
# Lazy model loading (memory optimization)
# -------------------------------
_model: Optional[EmotionLSTM] = None
_embedder: Optional[SentenceTransformer] = None

def get_model():
    """Lazy load model only when needed"""
    global _model
    if _model is None:
        try:
            _model = EmotionLSTM(
                input_dim=384,
                hidden_dim=128,
                num_classes=3
            )
            _model.load_state_dict(
                torch.load("data/processed/emotion_lstm.pt", map_location=torch.device('cpu'))
            )
            _model.eval()
            gc.collect()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")
    return _model

def get_embedder():
    """Lazy load embedder only when needed"""
    global _embedder
    if _embedder is None:
        try:
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            gc.collect()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Embedder loading failed: {str(e)}")
    return _embedder

# -------------------------------
# Request / Response schema
# -------------------------------
class ConversationRequest(BaseModel):
    conversation: List[str]

# -------------------------------
# Health check endpoints
# -------------------------------
@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "Conversation Sentiment API",
        "models_loaded": _model is not None and _embedder is not None
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# Handle CORS preflight requests
@app.options("/predict")
async def predict_options():
    return {}

@app.options("/health")
async def health_options():
    return {}

# -------------------------------
# Main prediction endpoint
# -------------------------------
@app.post("/predict")
def analyze_sentiment(request: ConversationRequest):
    try:
        sentences = request.conversation
        
        if not sentences:
            raise HTTPException(status_code=400, detail="No conversation data provided")
        
        # Lazy load models
        model = get_model()
        embedder = get_embedder()
        
        # Process in smaller batches to reduce memory usage
        embeddings = embedder.encode(sentences, batch_size=8, show_progress_bar=False)
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
        
        # Clean up tensors to free memory
        del X, logits, probs, embeddings
        gc.collect()
        
        return {"results": results}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)