import torch
from trainer import VAETrainer
from model import VAE
from config import get_default_config
from data_loader import get_data_loaders
from utils import Logger
import argparse

# Example toy data (replace with real SMILES data for production)
def get_toy_data():
    return [
        'CCO', 'CCN', 'CCC', 'CCCl', 'CCBr',
        'C1CC1', 'C1CCO1', 'C1CCN1', 'C1CCC1', 'C1CCCl1',
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    train_data = get_toy_data()
    train_loader, vocab = get_data_loaders(train_data, batch_size=args.batch_size)

    config = get_default_config()
    model = VAE(vocab, config).to(args.device)
    # trainer = VAETrainer(config)

    logger = Logger()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for batch in train_loader:
            batch = [b.to(args.device) for b in batch]
            kl_loss, recon_loss = model(batch)
            loss = kl_loss + recon_loss
            loss.backward()
            # (Add optimizer step, zero_grad, etc. for full training)
            logger.append({'kl_loss': kl_loss.item(), 'recon_loss': recon_loss.item(), 'loss': loss.item()})
        print(f"Loss: {loss.item():.4f}")
    logger.save('train_log.csv')

    # After training, print reconstructions and random samples
    model.eval()
    with torch.no_grad():
        print("\nOriginal vs Reconstruction:")
        for batch in train_loader:
            batch = [b.to(args.device) for b in batch]
            z, _ = model.forward_encoder(batch)
            lengths = [len(x) for x in batch]
            x_padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=model.pad)
            x_emb = model.x_emb(x_padded)
            z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
            x_input = torch.cat([x_emb, z_0], dim=-1)
            x_input = torch.nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True, enforce_sorted=False)
            h_0 = model.decoder_lat(z)
            h_0 = h_0.unsqueeze(0).repeat(model.decoder_rnn.num_layers, 1, 1)
            output, _ = model.decoder_rnn(x_input, h_0)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            y = model.decoder_fc(output)
            pred = y.argmax(-1)
            for inp, out in zip(batch, pred):
                print("IN :", model.tensor2string(inp))
                print("OUT:", model.tensor2string(out))
            break  # Only show one batch

        print("\nRandom samples from latent space:")
        samples = model.sample(5)
        for s in samples:
            print(s)

if __name__ == '__main__':
    main()
