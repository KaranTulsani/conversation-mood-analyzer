import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lstm_model import EmotionLSTM

# -------------------------------
# Load prepared tensors
# -------------------------------
X = torch.load("data/processed/X.pt")
y = torch.load("data/processed/y.pt")

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)

# -------------------------------
# Dataset & DataLoader
# -------------------------------
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -------------------------------
# Model
# -------------------------------
model = EmotionLSTM(
    input_dim=384,
    hidden_dim=128,
    num_classes=3
)

# -------------------------------
# Class weights (IMPORTANT CHANGE)
# -------------------------------
# Order: [neutral, anger, disgust, fear, happiness, sadness, surprise]
# Sentiment class weights
# [neutral, negative, positive]
class_weights = torch.tensor(
    [0.4,   # neutral (still common)
     1.2,   # negative
     1.0],  # positive
    dtype=torch.float32
)


# -------------------------------
# Training setup
# -------------------------------
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=-1
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------
# Training loop
# -------------------------------
epochs = 10

for epoch in range(epochs):
    total_loss = 0.0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        logits = model(X_batch)  # (B, T, 7)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y_batch.view(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# -------------------------------
# Save trained model
# -------------------------------
torch.save(model.state_dict(), "data/processed/emotion_lstm.pt")
print("✅ Model saved as emotion_lstm.pt")
