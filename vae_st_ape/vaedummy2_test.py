import torch
from vaedummy2 import VAEDummy2, DummyDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAEDummy2().to(device)
    dataset = DummyDataset(num_samples=8, seq_len=8, vocab_size=10)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            out_seq = torch.argmax(logits, dim=-1)
            print("Input:", batch.cpu().numpy())
            print("Recon:", out_seq.cpu().numpy())
        print("Sampled:", model.sample(2, device=device).cpu().numpy())
