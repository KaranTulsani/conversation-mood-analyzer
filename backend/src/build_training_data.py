import pandas as pd
import torch
import ast
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
import os
from sentiment_map import EMOTION_TO_SENTIMENT


# -------------------------------
# Load CLEAN data (IMPORTANT)
# -------------------------------
df = pd.read_csv("data/processed/clean_dialogs.csv")
print("Loaded clean data:", df.shape)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

X, y = [], []

for _, row in df.iterrows():
    turns = ast.literal_eval(row["dialog"])
    raw_emotions = ast.literal_eval(row["emotion"])
    emotions = [EMOTION_TO_SENTIMENT[e] for e in raw_emotions]

    embeddings = embedder.encode(turns)

    X.append(torch.tensor(embeddings, dtype=torch.float32))
    y.append(torch.tensor(emotions, dtype=torch.long))

# -------------------------------
# Pad sequences
# -------------------------------
X_padded = pad_sequence(X, batch_first=True)
y_padded = pad_sequence(y, batch_first=True, padding_value=-1)

print("X shape:", X_padded.shape)
print("y shape:", y_padded.shape)

# -------------------------------
# Save tensors
# -------------------------------
os.makedirs("data/processed", exist_ok=True)
torch.save(X_padded, "data/processed/X.pt")
torch.save(y_padded, "data/processed/y.pt")

print("✅ Saved X.pt and y.pt")
