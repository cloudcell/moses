"""
DO NOT USE THIS FILE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEDummy2(nn.Module):
    def __init__(self, vocab_size=10, emb_dim=16, hidden_dim=32, num_layers=1, max_len=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len] integer tokens
        emb = self.embedding(x)
        _, (h, c) = self.encoder(emb)
        # Decoder input: shift x right, prepend 0 (start token)
        dec_in = torch.zeros_like(x)
        dec_in[:, 1:] = x[:, :-1]
        dec_emb = self.embedding(dec_in)
        out, _ = self.decoder(dec_emb, (h, c))
        logits = self.fc_out(out)
        return logits

    def sample(self, batch_size=1, max_len=None, device=None):
        if max_len is None:
            max_len = self.max_len
        if device is None:
            device = next(self.parameters()).device
        # Start with start token 0
        inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        outputs = []
        for _ in range(max_len):
            emb = self.embedding(inputs[:, -1:])
            out, (h, c) = self.decoder(emb, (h, c))
            logits = self.fc_out(out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            outputs.append(next_token)
            inputs = torch.cat([inputs, next_token], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs

# Dummy dataset generator for overfitting
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, seq_len=8, vocab_size=10):
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Simple train loop for overfitting
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAEDummy2().to(device)
    dataset = DummyDataset(num_samples=32, seq_len=8, vocab_size=10)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(200):
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), batch.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch} loss: {loss.item():.4f}")
    # Test model on a sample
    model.eval()
    with torch.no_grad():
        sample = dataset[0].unsqueeze(0).to(device)
        out_logits = model(sample)
        out_seq = torch.argmax(out_logits, dim=-1)
        print("Input: ", sample.cpu().numpy())
        print("Recon: ", out_seq.cpu().numpy())
        print("Sampled: ", model.sample(1, device=device).cpu().numpy())
