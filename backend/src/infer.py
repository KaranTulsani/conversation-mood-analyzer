import torch
import torch.nn.functional as F
import re
from sentence_transformers import SentenceTransformer
from lstm_model import EmotionLSTM
from sentiment_map import SENTIMENT_LABELS

# -------------------------------
# Text normalization helper
# -------------------------------
def normalize_words(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation
    return set(text.split())

# -------------------------------
# Sentiment lexicons
# -------------------------------
NEGATIVE_WORDS = {
    "tired", "stress", "stressful", "sad", "angry", "upset",
    "exhausted", "bad", "terrible", "worried"
}

POSITIVE_WORDS = {
    "hope", "hoping", "improve", "happy", "good", "great",
    "better", "excited", "love"
}

# -------------------------------
# Load model (3 sentiment classes)
# -------------------------------
model = EmotionLSTM(
    input_dim=384,
    hidden_dim=128,
    num_classes=3
)

model.load_state_dict(
    torch.load("data/processed/emotion_lstm.pt")
)
model.eval()

# -------------------------------
# Embedder
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Test conversation
# -------------------------------
conversation = [
    "I'm not feeling very well.",
    "but I'm trying to be good.",
    "Work has been really stressful.",
    "But I'm hoping things improve soon."
]

embeddings = embedder.encode(conversation)
X = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)

# -------------------------------
# Predict (model + override)
# -------------------------------
with torch.no_grad():
    logits = model(X)
    probs = F.softmax(logits, dim=-1).squeeze(0)

final_preds = []

for i, sentence in enumerate(conversation):
    pred = torch.argmax(probs[i]).item()  # model prediction
    words = normalize_words(sentence)

    # Override ONLY if model says neutral
    if pred == 0:
        if words & NEGATIVE_WORDS:
            pred = 1
        elif words & POSITIVE_WORDS:
            pred = 2

    final_preds.append(pred)

# -------------------------------
# Print results
# -------------------------------
print("\nConversation Sentiment Analysis:\n")

for i, sentence in enumerate(conversation):
    sentiment = SENTIMENT_LABELS[final_preds[i]]
    print(f"{i+1}. {sentence}")
    print(f"   → Sentiment: {sentiment}\n")
