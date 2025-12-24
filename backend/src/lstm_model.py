import torch
import torch.nn as nn

class EmotionLSTM(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, num_classes=3):
        super().__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Fully connected layer to map LSTM output → emotion classes
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        """
        lstm_out, _ = self.lstm(x)

        # Apply classifier to each time step
        logits = self.fc(lstm_out)

        return logits
