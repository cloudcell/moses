import torch
from torch.utils.data import Dataset, DataLoader
from utils import OneHotVocab

class StringDataset(Dataset):
    def __init__(self, data, vocab=None):
        self.data = data
        if vocab is None:
            self.vocab = OneHotVocab.from_data(data)
        else:
            self.vocab = vocab
        self.tokens = [self.vocab.string2ids(s) for s in data]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.tokens[idx]

def get_data_loaders(train_data, batch_size=64, shuffle=True):
    train_dataset = StringDataset(train_data)
    vocab = train_dataset.vocab
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: [torch.tensor(seq, dtype=torch.long) for seq in batch]
    )
    return train_loader, vocab
