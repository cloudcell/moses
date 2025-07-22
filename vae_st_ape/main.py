import torch
from trainer import VAETrainer
from model import VAE, VAEDummy
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

DEBUG = False

def main():
    global DEBUG, model, single_train_batch, train_dataset, idx_to_token
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vocab_file', type=str, default=None, help='Path to APETokenizer vocab file (tokenizer.json). If not provided, will download from HuggingFace Hub.')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'vaedummy'], help='Which model to use: vae (default) or vaedummy (plain LSTM autoencoder)')
    parser.add_argument('--gen_vocab', action='store_true', help='Train tokenizer from data and save vocab (then exit)')
    parser.add_argument('--single_batch', action='store_true', help='Train and eval on a single batch for overfitting/debugging')
    parser.add_argument('--min_loss', type=float, default=0.1, help='Minimum loss threshold for early stopping')
    args = parser.parse_args()

    if args.gen_vocab:
        # Train tokenizer from all data and save vocab
        from apetokenizer.ape_tokenizer import APETokenizer
        import selfies
        train_smiles = load_smiles_file('data_trn.txt')
        val_smiles = load_smiles_file('data_val.txt')
        test_smiles = load_smiles_file('data_tst.txt')
        all_selfies = []
        skipped = 0
        for s in train_smiles + val_smiles + test_smiles:
            try:
                sf = selfies.encoder(s)
                all_selfies.append(sf)
            except selfies.exceptions.EncoderError as e:
                print(f"[WARN] Skipping SMILES not convertible to SELFIES: {s}\nReason: {e}")
                skipped += 1
        print(f"Total skipped SMILES: {skipped}")
        tokenizer = APETokenizer()
        print(f"Training tokenizer on {len(all_selfies)} molecules (SELFIES)...")
        tokenizer.train(all_selfies, type="selfies")
        out_path = args.vocab_file or "trained_vocab.json"
        tokenizer.save_vocabulary(out_path)
        print(f"Trained vocab saved to {out_path}")
        exit(0)

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

    if args.single_batch:
        args.epochs = 20
        args.batch_size = 4
        config = get_default_config()
        config.n_batch = 4
        DEBUG = args.debug

    # Load config
    config = get_default_config()
    # Load real dataset splits
    train_smiles = load_smiles_file('data_trn.txt')
    val_smiles = load_smiles_file('data_val.txt')
    test_smiles = load_smiles_file('data_tst.txt')
    train_loader, tokenizer = get_data_loaders(train_smiles, batch_size=config.n_batch, shuffle=True, tokenizer=tokenizer, config=config)
    val_loader, _ = get_data_loaders(val_smiles, batch_size=config.n_batch, shuffle=False, tokenizer=tokenizer, config=config)
    test_loader, _ = get_data_loaders(test_smiles, batch_size=config.n_batch, shuffle=False, tokenizer=tokenizer, config=config)

    if args.single_batch:
        # Use only the first batch for both train and val
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        single_train_batch = [next(train_iter)]
        single_val_batch = [next(val_iter)]
        class SingleBatchLoader:
            def __iter__(self):
                while True:
                    yield single_train_batch[0]
            def __len__(self):
                return 1
        class SingleValBatchLoader:
            def __iter__(self):
                while True:
                    yield single_val_batch[0]
            def __len__(self):
                return 1
        train_loader = SingleBatchLoader()
        val_loader = SingleValBatchLoader()

    # Diagnostic: print first batch of SMILES from both loaders
    print("\n[DIAGNOSTIC] Printing first batch of SMILES from train and val loaders:")
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    train_batch = next(train_iter)
    val_batch = next(val_iter)
    # Decode tokens back to SMILES for comparison
    from data_loader import StringDataset
    train_dataset = StringDataset(train_smiles, tokenizer=tokenizer)
    val_dataset = StringDataset(val_smiles, tokenizer=tokenizer)
    # --- Robust decoding: filter tokens not in SELFIES alphabet and <unk> 


    # Diagnostic: print first batch of SMILES from both loaders
    print("\n[DIAGNOSTIC] Printing first batch of SMILES from train and val loaders:")
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    train_batch = next(train_iter)
    val_batch = next(val_iter)
    # Decode tokens back to SMILES for comparison
    from data_loader import StringDataset
    train_dataset = StringDataset(train_smiles, tokenizer=tokenizer)
    val_dataset = StringDataset(val_smiles, tokenizer=tokenizer)
    # --- Robust decoding: filter tokens not in SELFIES alphabet and <unk> 
    import selfies
    selfies_alphabet = set(selfies.get_semantic_robust_alphabet())
    import selfies as selfies_lib
    def filter_tokens(token_list, vocab, selfies_alphabet, debug_first_seq=False):
        # Diagnostic: print the alphabet once
        if not hasattr(filter_tokens, 'printed_alphabet'):
            print('[DEBUG] selfies_alphabet:', selfies_alphabet)
            filter_tokens.printed_alphabet = True
        filtered = []
        for i, idx in enumerate(token_list):
            tok = vocab.get(idx, None) if isinstance(vocab, dict) else vocab[idx]
            if tok == '<unk>' or tok is None:
                continue
            if tok in ('<s>', '</s>'):
                continue
            # Split macro-token into atomic tokens and append each atomic token individually
            split_toks = selfies_lib.split_selfies(tok)
            if debug_first_seq:
                print(f'[DEBUG] filter_tokens: idx={i}, tok={tok!r}, split_toks={split_toks}')
            for atomic_tok in split_toks:
                if atomic_tok in selfies_alphabet:
                    filtered.append(atomic_tok)
                else:
                    if debug_first_seq:
                        print(f'[WARNING] atomic token {atomic_tok!r} not in alphabet!')
        if debug_first_seq:
            print(f'[DEBUG] filter_tokens: final filtered={filtered}')
        # Diagnostic: warn if filtered is empty
        if len(filtered) == 0:
            print(f'[DIAGNOSTIC] WARNING: filtered tokens are empty!')
            print(f'[DIAGNOSTIC] Original token_list: {token_list}')
            print(f'[DIAGNOSTIC] Original tokens: {[vocab.get(idx, None) if isinstance(vocab, dict) else vocab[idx] for idx in token_list]}')
        return filtered

    # Build idx_to_token as a list mapping index to token string
    vocab_size = len(tokenizer.vocabulary)
    idx_to_token = [None] * vocab_size
    for token, idx in tokenizer.vocabulary.items():
        idx_to_token[idx] = token
    print('[DIAGNOSTIC] First 10 idx_to_token:', idx_to_token[:10])
    # Filter and decode
    def robust_decode(batch, dataset, idx_to_token):
        decoded_raw = []
        decoded_filtered = []
        for i, seq in enumerate(batch):
            tokens = seq.tolist()
            try:
                raw = dataset.decode_tokens(tokens)
            except Exception as e:
                raw = f"[DecoderError: {e}]"
            # For the first sequence only, enable detailed debug in filter_tokens
            debug_first_seq = (i == 0)
            filtered = filter_tokens(tokens, idx_to_token, selfies_alphabet, debug_first_seq=debug_first_seq)
            selfies_str = ''.join(filtered)
            if i < 5:
                print(f"[DEBUG] robust_decode entry {i}: tokens={tokens}")
                print(f"[DEBUG] robust_decode entry {i}: filtered tokens={filtered}")
                print(f"[DEBUG] robust_decode entry {i}: selfies_str='{selfies_str}'")
            try:
                filtered_decoded = selfies.decoder(selfies_str)
            except Exception as e:
                filtered_decoded = f"[DecoderError: {e}]"
            if i < 5:
                print(f"[DEBUG] robust_decode entry {i}: decoded='{filtered_decoded}'")
            decoded_raw.append(raw)
            decoded_filtered.append(filtered_decoded)
        return decoded_raw, decoded_filtered


    train_raw, train_filtered = robust_decode(train_batch, train_dataset, idx_to_token)
    val_raw, val_filtered = robust_decode(val_batch, val_dataset, idx_to_token)
    print("Train batch SMILES (raw):", train_raw)
    print("Train batch SMILES (filtered):", train_filtered)
    print("Val batch SMILES (raw):", val_raw)
    print("Val batch SMILES (filtered):", val_filtered)
    print("Are filtered batches identical?", train_filtered == val_filtered)

    config = get_default_config()
    # --- Build model ---
    if args.model_type == 'vaedummy':
        print('Using VAEDummy (plain LSTM autoencoder)')
        model = VAEDummy(tokenizer, config)
    else:
        model = VAE(tokenizer, config)
    model = model.to(args.device)
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
    trainer.fit(model, train_loader, val_loader, logger=logger, epochs=args.epochs, lr=config.lr_start, scheduler=scheduler, checkpoint_dir=checkpoint_dir, start_epoch=start_epoch, min_loss=args.min_loss)
     
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
        n_samples = min(1000, len(test_smiles))
        random_indices = range(n_samples)
        with torch.no_grad():
            # use tqdm
            for idx in tqdm(random_indices):
                s = test_smiles[idx]
                input_tensor = model.string2tensor(s, device=args.device).unsqueeze(0)
                # Assert every token is a known token
                input_ids = input_tensor.squeeze(0).tolist()
                vocab_ids = set(range(len(tokenizer)))
                for token_id in input_ids:
                    assert token_id in vocab_ids, f"Unknown token_id {token_id} in input: {input_ids} (vocab size: {len(tokenizer)})"
                with torch.no_grad():
                    z, _ = model.forward_encoder(input_tensor)
                    out_tokens = model.sample(1, max_len=len(s)+10, z=z)  # returns list of SMILES via new pipeline
                    out = out_tokens[0]  # already SMILES via tensor2string
                valid = is_valid_smiles(out)
                log_f.write(f"IN : {s}\n")
                same = False
                mol_in = Chem.MolFromSmiles(s)
                mol_out = Chem.MolFromSmiles(out)                
                if mol_in is not None and mol_out is not None:
                    can_in = Chem.MolToSmiles(mol_in, canonical=True)
                    can_out = Chem.MolToSmiles(mol_out, canonical=True)
                    if can_in == can_out:
                        valid_reconstructions_cnt += 1
                        same = True
                status = 'valid & same' if valid and same else ('valid' if valid else 'invalid')
                log_f.write(f"OUT: {out}\t{status}\n")

                if valid:
                    valid_smiles_cnt += 1

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

    # After training: if single_batch, print input and reconstructions
    import sys
    if '--single_batch' in sys.argv:
        try:
            print("\n[SINGLE BATCH DEBUG] Printing input and reconstructed SMILES after training:")
            model.eval()
            with torch.no_grad():
                batch = single_train_batch[0]
                # Move to device if needed
                if hasattr(batch[0], 'to'):
                    batch = [b.to(model.x_emb.weight.device) for b in batch]
                # Get reconstructions
                if hasattr(model, 'tensor2string'):
                    recons = [model.tensor2string(b) for b in batch]
                else:
                    # fallback: try robust_decode
                    from main import robust_decode
                    _, recons = robust_decode(batch, train_dataset, idx_to_token)
                # Get original SMILES
                origs = [train_dataset.decode_tokens(b.tolist()) for b in batch]
                print(f"{'Original':<32} | {'Reconstructed':<32}")
                print('-'*68)
                for orig, recon in zip(origs, recons):
                    print(f"{orig:<32} | {recon:<32}")
                # Debug: print first 5 input/output token sequences and their decoded SMILES
                print("\n[DEBUG] First 5 input/output token sequences and decoded SMILES:")
                for i in range(min(5, len(batch))):
                    input_tokens = batch[i].tolist()
                    output_tokens = None
                    # Try to get output tokens from model if possible
                    if hasattr(model, 'sample'):
                        with torch.no_grad():
                            z, _ = model.forward_encoder(batch[i].unsqueeze(0))
                            out_tensor = model.sample(1, max_len=len(batch[i]), z=z, return_tensor=True)[0]
                        output_tokens = out_tensor.tolist()
                    else:
                        output_tokens = None
                    print(f"[DEBUG] Input {i}: tokens={input_tokens}")
                    print(f"[DEBUG] Input {i}: decoded SMILES={origs[i]}")
                    if output_tokens is not None:
                        print(f"[DEBUG] Output {i}: tokens={output_tokens}")
                        # Decode output tokens to SMILES
                        if hasattr(model, 'tensor2string'):
                            decoded_out = model.tensor2string(out_tensor)
                            print(f"[DEBUG] Output {i}: decoded SMILES={decoded_out}")
                        else:
                            print(f"[DEBUG] Output {i}: decoded SMILES={recons[i]}")
                    else:
                        print(f"[DEBUG] Output {i}: tokens=N/A")
        except Exception as e:
            print("[SINGLE BATCH DEBUG] Error printing reconstructions:", e)
