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
    # Simple train/val split for toy data
    split_idx = int(0.8 * len(train_data))
    train_smiles = train_data[:split_idx]
    val_smiles = train_data[split_idx:]
    train_loader, vocab = get_data_loaders(train_smiles, batch_size=args.batch_size)
    val_loader, _ = get_data_loaders(val_smiles, batch_size=args.batch_size, vocab=vocab)

    config = get_default_config()
    model = VAE(vocab, config).to(args.device)
    # trainer = VAETrainer(config)

    logger = Logger()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_start)
    from misc import KLAnnealer, CosineAnnealingLRWithRestart
    kl_annealer = KLAnnealer(args.epochs, config)
    lr_scheduler = CosineAnnealingLRWithRestart(optimizer, config)
    import os
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        kl_weight = kl_annealer(epoch)
        print(f"Epoch {epoch+1}/{args.epochs} (kl_weight={kl_weight:.4f}, lr={optimizer.param_groups[0]['lr']:.6f})")
        # --- Training ---
        model.train()
        for batch in train_loader:
            batch = [b.to(args.device) for b in batch]
            optimizer.zero_grad()
            kl_loss, recon_loss = model(batch)
            loss = kl_weight * kl_loss + recon_loss
            loss.backward()
            optimizer.step()
            logger.append({'kl_loss': kl_loss.item(), 'recon_loss': recon_loss.item(), 'loss': loss.item(), 'kl_weight': kl_weight, 'lr': optimizer.param_groups[0]['lr'], 'mode': 'train'})
        lr_scheduler.step()
        print(f"Train Loss: {loss.item():.4f}")
        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(args.device) for b in batch]
                kl_loss, recon_loss = model(batch)
                val_loss = kl_weight * kl_loss + recon_loss
                val_losses.append(val_loss.item())
                logger.append({'kl_loss': kl_loss.item(), 'recon_loss': recon_loss.item(), 'loss': val_loss.item(), 'kl_weight': kl_weight, 'lr': optimizer.param_groups[0]['lr'], 'mode': 'val'})
        mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('nan')
        print(f"Val Loss: {mean_val_loss:.4f}")
        # --- Checkpoint ---
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': mean_val_loss,
            'config': config.__dict__,
        }, checkpoint_path)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': mean_val_loss,
                'config': config.__dict__,
            }, os.path.join(checkpoint_dir, "best_model.pt"))
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
