import torch
from lstm_model import EmotionLSTM

# Create model
model = EmotionLSTM()

# Fake input: batch_size=1, seq_len=5, embedding_dim=384
dummy_input = torch.randn(1, 5, 384)

# Forward pass
output = model(dummy_input)

print("Output shape:", output.shape)
