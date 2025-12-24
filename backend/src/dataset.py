import torch
from torch.utils.data import Dataset

class ConversationDataset(Dataset):
    def __init__(self, embeddings, emotions):
        self.embeddings = embeddings
        self.emotions = emotions

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        y = torch.tensor(self.emotions[idx], dtype=torch.long)
        return x, y
