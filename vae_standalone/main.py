import torch
from trainer import VAETrainer
from model import VAE
from config import get_default_config
from data_loader import get_data_loaders
from utils import Logger
import argparse
from tqdm import tqdm
from rdkit import Chem

# Load SMILES from a text file (one per line)
def load_smiles_file(path, limit=None):
    smiles = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            s = line.strip()
            if s:
                smiles.append(s)
    return smiles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Set thread count if using CPU
    if args.device == 'cpu':
        torch.set_num_threads(126)

    # Load real dataset splits
    train_smiles = load_smiles_file('data_trn.txt')
    val_smiles = load_smiles_file('data_val.txt')
    test_smiles = load_smiles_file('data_tst.txt')
    train_loader, vocab = get_data_loaders(train_smiles, batch_size=args.batch_size)
    val_loader, _ = get_data_loaders(val_smiles, batch_size=args.batch_size, vocab=vocab)
    test_loader, _ = get_data_loaders(test_smiles, batch_size=args.batch_size, vocab=vocab)

    config = get_default_config()
    model = VAE(vocab, config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_start)
    from misc import KLAnnealer, CosineAnnealingLRWithRestart
    kl_annealer = KLAnnealer(args.epochs, config)
    lr_scheduler = CosineAnnealingLRWithRestart(optimizer, config)
    import os, glob
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    # --- Checkpoint loading ---
    start_epoch = 0
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pt')),
                              key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        print(f"Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")

    logger = Logger()
    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
        kl_weight = kl_annealer(epoch)
        print(f"Epoch {epoch+1}/{end_epoch} (kl_weight={kl_weight:.4f}, lr={optimizer.param_groups[0]['lr']:.6f})")
        # --- Training ---
        model.train()
        train_iter = tqdm(train_loader, desc=f'Train (epoch {epoch+1})', leave=False)
        for batch in train_iter:
            batch = [b.to(args.device) for b in batch]
            optimizer.zero_grad()
            kl_loss, recon_loss = model(batch)
            loss = kl_weight * kl_loss + recon_loss
            loss.backward()
            optimizer.step()
            logger.append({'kl_loss': kl_loss.item(), 'recon_loss': recon_loss.item(), 'loss': loss.item(), 'kl_weight': kl_weight, 'lr': optimizer.param_groups[0]['lr'], 'mode': 'train'})
            train_iter.set_postfix(loss=loss.item(), kl=kl_loss.item(), recon=recon_loss.item())
        lr_scheduler.step()
        print(f"Train Loss: {loss.item():.4f}")
        # --- Validation ---
        model.eval()
        val_losses = []
        val_iter = tqdm(val_loader, desc=f'Val (epoch {epoch+1})', leave=False)
        with torch.no_grad():
            for batch in val_iter:
                batch = [b.to(args.device) for b in batch]
                kl_loss, recon_loss = model(batch)
                val_loss = kl_weight * kl_loss + recon_loss
                val_losses.append(val_loss.item())
                logger.append({'kl_loss': kl_loss.item(), 'recon_loss': recon_loss.item(), 'loss': val_loss.item(), 'kl_weight': kl_weight, 'lr': optimizer.param_groups[0]['lr'], 'mode': 'val'})
                val_iter.set_postfix(loss=val_loss.item(), kl=kl_loss.item(), recon=recon_loss.item())
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

    # After training, evaluate on test set
    print("\nTest set evaluation:")
    model.eval()
    test_losses = []
    test_iter = tqdm(test_loader, desc='Test', leave=False)
    # Determine kl_weight for test (use last epoch or default 1.0)
    if args.epochs > 0:
        test_kl_weight = kl_annealer(args.epochs-1)
    else:
        test_kl_weight = 1.0
    
    if 0:
        with torch.no_grad():
            for batch in test_iter:
                batch = [b.to(args.device) for b in batch]
                kl_loss, recon_loss = model(batch)
                test_loss = test_kl_weight * kl_loss + recon_loss
                test_losses.append(test_loss.item())
                test_iter.set_postfix(loss=test_loss.item(), kl=kl_loss.item(), recon=recon_loss.item())
        mean_test_loss = sum(test_losses) / len(test_losses) if test_losses else float('nan')
        print(f"Test Loss: {mean_test_loss:.4f}")

    import random
    def is_valid_smiles(smiles):
        from rdkit import Chem
        import rdkit
        from contextlib import contextmanager
        import sys, os
        @contextmanager
        def suppress_stderr():
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stderr = old_stderr
        with suppress_stderr():
            m = Chem.MolFromSmiles(smiles)
        return m is not None


    # also, count the number of valid reconstructions
    valid_reconstructions_cnt = 0
    valid_smiles_cnt = 0
    print("\nReconstructions for 100 random test samples:")
    random_indices = random.sample(range(len(test_smiles)), 100)
    with torch.no_grad():
        for idx in random_indices:
            s = test_smiles[idx]
            input_tensor = model.string2tensor(s, device=args.device).unsqueeze(0)
            # Encode to latent z, then decode for reconstruction
            with torch.no_grad():
                z, _ = model.forward_encoder([input_tensor.squeeze(0)])
                out = model.sample(1, max_len=len(s)+5, z=z)
            print(f"IN : {s}")
            print(f"OUT: {out[0]}", end='')
            print("\tvalid" if is_valid_smiles(out[0]) else "\tinvalid")
            if is_valid_smiles(out[0]):
                valid_smiles_cnt += 1
            # if in == out, count as valid reconstruction
            if s == out[0]:
                valid_reconstructions_cnt += 1
    print(f"Valid smiles: {valid_smiles_cnt}/{100}")
    print(f"Valid reconstructions: {valid_reconstructions_cnt}/{100}")

if __name__ == '__main__':
    main()
