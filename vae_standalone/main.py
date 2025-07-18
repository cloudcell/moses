import torch
from trainer import VAETrainer
from model import VAE
from config import get_default_config
from data_loader import get_data_loaders
from utils import Logger
import argparse
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# import random
import numpy as np

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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set thread count if using CPU
    if args.device == 'cpu':
        torch.set_num_threads(126)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # random.seed(args.seed)
    np.random.seed(args.seed)

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
    if 0:
        test_losses = []
        test_iter = tqdm(test_loader, desc='Test', leave=False)
        # Determine kl_weight for test (use last epoch or default 1.0)
        if args.epochs > 0:
            test_kl_weight = kl_annealer(args.epochs-1)
        else:
            test_kl_weight = 1.0
    
        with torch.no_grad():
            for batch in test_iter:
                batch = [b.to(args.device) for b in batch]
                kl_loss, recon_loss = model(batch)
                test_loss = test_kl_weight * kl_loss + recon_loss
                test_losses.append(test_loss.item())
                test_iter.set_postfix(loss=test_loss.item(), kl=kl_loss.item(), recon=recon_loss.item())
        mean_test_loss = sum(test_losses) / len(test_losses) if test_losses else float('nan')
        print(f"Test Loss: {mean_test_loss:.4f}")

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
    edit_distance_sum = 0
    edit_distance_min = float('inf')
    edit_distance_max = float('-inf')
    try:
        import editdistance
        def edit_distance(a, b):
            return editdistance.eval(a, b)
    except ImportError:
        from difflib import SequenceMatcher
        def edit_distance(a, b):
            # Approximate Levenshtein distance using SequenceMatcher
            # (not exactly the same, but gives an idea if editdistance is not installed)
            sm = SequenceMatcher(None, a, b)
            return max(len(a), len(b)) - int(sm.ratio() * max(len(a), len(b)))

    import datetime
    log_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(log_dir, exist_ok=True)
    current_epoch = start_epoch if 'start_epoch' in locals() else 0
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"epoch_{current_epoch}_{now}.log"
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, 'w') as log_f:
        log_f.write(f"Reconstructions for 1000 first test samples (epoch {current_epoch}):\n")
        random_indices = range(1000)
        with torch.no_grad():
            # use tqdm
            for idx in tqdm(random_indices):
                s = test_smiles[idx]
                input_tensor = model.string2tensor(s, device=args.device).unsqueeze(0)
                with torch.no_grad():
                    z, _ = model.forward_encoder(input_tensor)
                    out = model.sample(1, max_len=len(s)+10, z=z)  # contains .forward_decoder and returns strings
                    out = out[0]
                valid = is_valid_smiles(out)
                log_f.write(f"IN : {s}\n")
                log_f.write(f"OUT: {out}\t{'valid' if valid else 'invalid'}\n")
                if valid:
                    valid_smiles_cnt += 1

                # check if the reconstruction is the same as the input using both canonical smiles
                if Chem.MolFromSmiles(s) == Chem.MolFromSmiles(out):
                    valid_reconstructions_cnt += 1

                # calculate the edit distance
                edit_dist = edit_distance(s, out)
                edit_distance_sum += edit_dist
                edit_distance_min = min(edit_distance_min, edit_dist)
                edit_distance_max = max(edit_distance_max, edit_dist)
                log_f.write(f"Edit distance: {edit_dist}\n")

                # update tqdm string
                # tqdm.write(f"Valid smiles: {valid_smiles_cnt}, Valid reconstructions: {valid_reconstructions_cnt}/1000, Total Edit distance: {edit_distance_sum}")

        log_f.write(f"Valid smiles: {valid_smiles_cnt}/1000\n")
        log_f.write(f"Valid reconstructions: {valid_reconstructions_cnt}/1000\n")
        log_f.write(f"Total Edit distance: {edit_distance_sum}\n")
        log_f.write(f"Min Edit distance: {edit_distance_min}\n")
        log_f.write(f"Max Edit distance: {edit_distance_max}\n")

    print(f"Test results logged to {log_path}")
    print(f"Valid smiles: {valid_smiles_cnt}/1000")
    print(f"Valid reconstructions: {valid_reconstructions_cnt}/1000")
    print(f"Total Edit distance: {edit_distance_sum}")
    print(f"Min Edit distance: {edit_distance_min}")
    print(f"Max Edit distance: {edit_distance_max}")

if __name__ == '__main__':
    main()
