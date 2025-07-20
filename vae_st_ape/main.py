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
import datetime as dt

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
    parser.add_argument('--vocab_file', type=str, default=None, help='Path to APETokenizer vocab file (tokenizer.json). If not provided, will download from HuggingFace Hub.')
    args = parser.parse_args()

    # --- Load APETokenizer vocab ---
    from apetokenizer.ape_tokenizer import APETokenizer
    if args.vocab_file is not None:
        vocab_path = args.vocab_file
        print(f"Loading APETokenizer vocab from file: {vocab_path}")
    else:
        import os
        local_dir = os.path.join(os.path.dirname(__file__), 'downloaded')
        os.makedirs(local_dir, exist_ok=True)
        local_vocab_path = os.path.join(local_dir, 'tokenizer.json')
        if os.path.exists(local_vocab_path):
            print(f"Found existing vocab at: {local_vocab_path}")
            vocab_path = local_vocab_path
        else:
            print("No vocab file supplied, downloading from HuggingFace Hub...")
            from huggingface_hub import hf_hub_download
            import shutil
            vocab_path = hf_hub_download(
                repo_id="mikemayuare/SELFYAPE",
                filename="tokenizer.json",
            )
            shutil.copy(vocab_path, local_vocab_path)
            print(f"Downloaded vocab to: {vocab_path}")
            print(f"Copied vocab to local path: {local_vocab_path}")
            vocab_path = local_vocab_path
    from load_hf_vocab import extract_hf_vocab
    tokenizer = APETokenizer()
    # Use wrapper to extract vocab if needed, then set directly
    vocab_dict = extract_hf_vocab(vocab_path)
    tokenizer.vocabulary = vocab_dict
    tokenizer.update_reverse_vocabulary()
    print("Vocabulary size:", len(tokenizer.vocabulary))

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
    train_loader, _ = get_data_loaders(train_smiles, batch_size=args.batch_size, tokenizer=tokenizer)
    val_loader, _ = get_data_loaders(val_smiles, batch_size=args.batch_size, tokenizer=tokenizer, shuffle=False)
    test_loader, _ = get_data_loaders(test_smiles, batch_size=args.batch_size, tokenizer=tokenizer, shuffle=False)

    config = get_default_config()
    model = VAE(tokenizer, config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_start)
    from misc import KLAnnealer
    kl_annealer = KLAnnealer(args.epochs, config)
    import os, glob
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    # --- Checkpoint loading ---
    start_epoch = 0
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pt')),
                              key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    scheduler = None  # will be created after optimizer
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        print(f"Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Restore scheduler after creation below
        metadata = checkpoint.get('metadata', {})
        start_epoch = metadata.get('epoch', checkpoint.get('epoch', 0))
        best_val_loss = metadata.get('val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    else:
        metadata = {}

    logger = Logger()
    end_epoch = start_epoch + args.epochs
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience, min_lr=config.lr_end)
    # Restore scheduler state if present
    if 'scheduler_state_dict' in metadata:
        scheduler.load_state_dict(metadata['scheduler_state_dict'])
    trainer = VAETrainer(config)
    trainer.fit(model, train_loader, val_loader, logger=logger, epochs=args.epochs, lr=config.lr_start, scheduler=scheduler, checkpoint_dir=checkpoint_dir, start_epoch=start_epoch)
     
    # save log with a timestamp
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.save(f'train_log_{timestamp}.csv')

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

    log_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(log_dir, exist_ok=True)
    current_epoch = start_epoch if 'start_epoch' in locals() else 0
    # Make epoch numbering 1-based and zero-padded to 5 digits
    epoch_str = f"{current_epoch+1:05d}"
    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"epoch_{epoch_str}_{now}.log"
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
                    out_tokens = model.sample(1, max_len=len(s)+10, z=z)  # returns list of SMILES via new pipeline
                    out = out_tokens[0]  # already SMILES via tensor2string
                valid = is_valid_smiles(out)
                log_f.write(f"IN : {s}\n")
                log_f.write(f"OUT: {out}\t{'valid' if valid else 'invalid'}\n")
                if valid:
                    valid_smiles_cnt += 1

                # check if the reconstruction is the same as the input using canonical SMILES (chemical equivalence)
                mol_in = Chem.MolFromSmiles(s)
                mol_out = Chem.MolFromSmiles(out)
                if mol_in is not None and mol_out is not None:
                    can_in = Chem.MolToSmiles(mol_in, canonical=True)
                    can_out = Chem.MolToSmiles(mol_out, canonical=True)
                    if can_in == can_out:
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
