import os
print("[DEBUG] main.py loaded and running")
import torch
from trainer import VAETrainer
from model import VAE, VAEDummy, VAEDummy2, VAENovo
from config import get_default_config, get_vaedummy_config
from data_loader import get_data_loaders
from utils import Logger
import argparse
from tqdm import tqdm

from rdkit import Chem, RDLogger
import numpy as np
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.nn.functional as F
import inspect
import sys
import selfies
from apetokenizer.ape_tokenizer import APETokenizer
import collections

import shutil
import random
from huggingface_hub import hf_hub_download
from load_hf_vocab import extract_hf_vocab

RDLogger.DisableLog('rdApp.*')

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


# --- Model checkpoint saving at end of training ---
def save_final_checkpoint(model, optimizer, epoch, val_loss, config):
    # Use config.start_time as timestamp for the folder
    timestamp = getattr(config, 'start_time', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print(f"[DEBUG] save_final_checkpoint: config.start_time = {getattr(config, 'start_time', None)}")
    ckpt_dir = os.path.join(os.path.dirname(__file__), f'checkpoints_{timestamp}')
    print(f"[DEBUG] Intended checkpoint directory: {ckpt_dir}")
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'model_final.pt')
        print(f"[DEBUG] Saving checkpoint to {ckpt_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': {
                'epoch': epoch,
                'val_loss': val_loss,
                'config': vars(config)
            }
        }, ckpt_path)
        print(f"[CHECKPOINT] Final model saved to {ckpt_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")



def main():
    global DEBUG, model, single_train_batch, train_dataset, idx_to_token
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vocab_file', type=str, default=None, help='Path to APETokenizer vocab file (tokenizer.json). If not provided, will download from HuggingFace Hub.')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'vaedummy', 'vaedummy2', 'vaenovo'], help='Which model to use: vae (default), vaedummy (plain LSTM autoencoder), vaedummy2 (integer LSTM), or vaenovo (minimal LSTM VAE)')
    parser.add_argument('--gen_vocab', action='store_true', help='Train tokenizer from data and save vocab (then exit)')
    parser.add_argument('--single_batch', action='store_true', help='Train and eval on a single batch for overfitting/debugging')
    parser.add_argument('--min_loss', type=float, default=0.1, help='Minimum loss threshold for early stopping')
    args = parser.parse_args()

    def edit_distance(a, b):
        dp = np.zeros((len(a)+1, len(b)+1), dtype=int)
        for i in range(len(a)+1): dp[i,0]=i
        for j in range(len(b)+1): dp[0,j]=j
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                if a[i-1]==b[j-1]:
                    dp[i,j]=dp[i-1,j-1]
                else:
                    dp[i,j]=1+min(dp[i-1,j],dp[i,j-1],dp[i-1,j-1])
        return dp[len(a),len(b)]

    # Canonicalize SMILES using RDKit
    def canonicalize(sm):
        import rdkit.Chem as Chem
        try:
            m = Chem.MolFromSmiles(sm)
            if m is None:
                return ''
            return Chem.MolToSmiles(m, canonical=True)
        except Exception:
            return ''

    
    if args.model_type == 'vaedummy2':

        
        cfg = get_vaedummy_config()

        print("[INFO] Using VAEDummy2: LSTM model for SMILES→SELFIES→tokens (APETokenizer)→reconstruction.")
        # --- Logging config and args ---

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = "test_results"
        os.makedirs(results_dir, exist_ok=True)
        run_log_path = os.path.join(results_dir, f"run_log_{timestamp}.txt")
        with open(run_log_path, 'w') as flog:
            flog.write("# VAEDummy2 Run Log\n")
            flog.write(f"Timestamp: {timestamp}\n")
            flog.write("\n[Config]\n")
            for k, v in vars(cfg).items():
                flog.write(f"{k}: {v}\n")
            flog.write("\n[Command Line]\n")
            flog.write(' '.join(sys.argv) + "\n")
        # Save path for later appending
        run_log_append_path = run_log_path


        # Require vocab file to be found at cfg.vocab_file (if not, exit)
        if not os.path.exists(cfg.vocab_file):
            print("[ERROR] vocab_file must exist for vaedummy2 with real APETokenizer vocab.")
            sys.exit(1)

        print(f"[INFO] Loading APETokenizer vocab from {cfg.vocab_file}")
        tokenizer = APETokenizer()
        tokenizer.load_vocabulary(cfg.vocab_file)
        print(f"[INFO] Loaded vocab size: {len(tokenizer)}")
        # Load SMILES from data files
        train_smiles = load_smiles_file('data_trn.txt')
        val_smiles = load_smiles_file('data_val.txt')
        test_smiles = load_smiles_file('data_tst.txt')
        print(f"[INFO] Loaded {len(train_smiles)} train, {len(val_smiles)} val, {len(test_smiles)} test molecules.")

        # Filter invalid SMILES using RDKit
        from rdkit import Chem
       
        def is_valid_smiles(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        n_train0, n_val0, n_test0 = len(train_smiles), len(val_smiles), len(test_smiles)
        train_smiles = [s for s in train_smiles if is_valid_smiles(s)]
        val_smiles = [s for s in val_smiles if is_valid_smiles(s)]
        test_smiles = [s for s in test_smiles if is_valid_smiles(s)]
        print(f"[INFO] Filtered: {n_train0-len(train_smiles)} invalid train, {n_val0-len(val_smiles)} invalid val, {n_test0-len(test_smiles)} invalid test SMILES.")

        # For fast debug, use only first 1000 train/val/test
        if 0:
            N = 1000
            train_smiles = train_smiles[:N]
            val_smiles = val_smiles[:N]
            test_smiles = test_smiles[:N]
        else:
            train_smiles = train_smiles
            val_smiles = val_smiles
            test_smiles = test_smiles

        vocab_size = len(tokenizer)
        device = torch.device(args.device)
        # model = VAEDummy2(vocab_size=vocab_size, emb_dim=256*2, hidden_dim=128 * 2, num_layers=2, max_len=24).to(device)  # 442
        # model = VAEDummy2(vocab_size=vocab_size, emb_dim=1024, hidden_dim=128 * 2, num_layers=2, max_len=24).to(device)  # 442 -- main
        model = VAEDummy2(vocab_size=vocab_size, 
                          emb_dim=cfg.emb_dim, 
                          hidden_dim=cfg.hidden_dim, 
                          num_layers_enc=cfg.num_layers_enc,
                          num_layers_dec=cfg.num_layers_dec, 
                          max_len=cfg.max_len, 
                          enc_dropout=cfg.enc_dropout, 
                          dec_dropout=cfg.dec_dropout,
                          use_lstm=cfg.use_lstm
                          )
        if torch.cuda.device_count() > 1:
            print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel.")
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        
        # Prepare tokenized training set
        string2tensor_fn = model.module.string2tensor if hasattr(model, 'module') else model.string2tensor
        token_tensors = [string2tensor_fn(s, tokenizer, device=device) for s in train_smiles]
        max_len = max(t.size(0) for t in token_tensors)

        padded = torch.full((len(token_tensors), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        for i, t in enumerate(token_tensors):
            padded[i, :t.size(0)] = t
        loader = torch.utils.data.DataLoader(padded, batch_size=args.batch_size, shuffle=True, drop_last=True)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_start)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.lr_end)
        tqdm_bar_args = dict(leave=True, ascii=True, ncols=100, dynamic_ncols=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f'runs/vaedummy2_{timestamp}')
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch}", **tqdm_bar_args)
            for batch_idx, batch in enumerate(pbar):
                logits, z = model(batch)
                loss = F.cross_entropy(logits.view(-1, vocab_size), batch.view(-1), ignore_index=tokenizer.pad_token_id)
                opt.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += loss.item() * batch.size(0)
                avg_loss = epoch_loss / ((batch_idx+1) * batch.size(0))
                pbar.set_postfix(loss=f"{avg_loss:.8f}")
            epoch_loss /= len(loader.dataset)
            scheduler.step(epoch_loss)
            current_lr = opt.param_groups[0]['lr']
            print(f"[LR] Epoch {epoch+1}: lr={current_lr:.8f}")
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
            writer.add_scalar('Loss/best_val', best_val_loss, epoch)
            if epoch_loss < args.min_loss:
                print(f"[EARLY STOP] Stopping training at epoch {epoch+1} due to min_loss criterion: {epoch_loss:.8f} < {args.min_loss}")
                break
        writer.close()
        # Evaluate on test set, save results
        model.eval()
        end_epoch = epoch + 1  # Track number of epochs actually run

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = "test_results"
        # os.makedirs(results_dir, exist_ok=True)
        out_path = run_log_path

        with torch.no_grad(), open(out_path, 'a') as f:  
            #, open(run_log_append_path, 'a') as flog:
            stats = collections.Counter()
            edit_distances = []
            can_edit_distances = []
            f.write("# VAEDummy2 Test Results\n")
            # flog.write("\n# VAEDummy2 Test Results\n")
            for i, s in enumerate(tqdm(test_smiles, desc="Testing", **tqdm_bar_args)):
                string2tensor_fn = model.module.string2tensor if hasattr(model, 'module') else model.string2tensor
                t = string2tensor_fn(s, tokenizer, device=device).unsqueeze(0)
                out_logits, _ = model(t)
                out_ids = torch.argmax(out_logits, dim=-1)[0]
                recon_smiles = model.tensor2string(out_ids, tokenizer)
                input_tokens = t.squeeze(0).tolist()
                recon_tokens = out_ids.tolist()

                ed = edit_distance(s, recon_smiles)
                can_s = canonicalize(s)
                can_recon = canonicalize(recon_smiles)
                can_ed = edit_distance(can_s, can_recon)
                # Validity check
                valid = (can_recon != '')
                exact = (recon_smiles == s)
                can_exact = (can_recon == can_s) and (can_recon != '')
                stats['total'] += 1
                if valid: stats['valid'] += 1
                if exact: stats['exact'] += 1
                if can_exact: stats['can_exact'] += 1
                edit_distances.append(ed)
                can_edit_distances.append(can_ed)
                # Write as row-oriented block
                f.write(f"---\n")
                f.write(f"InputSMILES: {s}\n")
                f.write(f"ReconSMILES: {recon_smiles}\n")
                f.write(f"InputTokens: {input_tokens}\n")
                f.write(f"ReconTokens: {recon_tokens}\n")
                f.write(f"EditDistance: {ed}\n")
                f.write(f"CanInputSMILES: {can_s}\n")
                f.write(f"CanReconSMILES: {can_recon}\n")
                f.write(f"CanEditDistance: {can_ed}\n")
                f.write(f"Valid: {valid}\n")
                f.write(f"Exact: {exact}\n")
                f.write(f"CanExact: {can_exact}\n")
            # Compile and write statistics
            mean_ed = sum(edit_distances)/len(edit_distances) if edit_distances else 0.0
            mean_can_ed = sum(can_edit_distances)/len(can_edit_distances) if can_edit_distances else 0.0
            stats_block = (
                "===\n"
                f"Total: {stats['total']}\n"
                f"Valid: {stats['valid']}\n"
                f"Exact: {stats['exact']}\n"
                f"CanExact: {stats['can_exact']}\n"
                f"MeanEditDistance: {mean_ed:.3f}\n"
                f"MeanCanEditDistance: {mean_can_ed:.3f}\n"
            )
            f.write(stats_block)
            # flog.write(stats_block)
        print(f"[INFO] Test results saved to {out_path}\n[INFO] Run log saved to {run_log_append_path}")
        print("[DEBUG] About to save checkpoint before return (VAEDummy2)")
        save_final_checkpoint(model, opt, end_epoch, best_val_loss, cfg)
        return

    if args.gen_vocab:
        # Train tokenizer from all data and save vocab
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
        print(f"[INFO] Tokenizer saved to {out_path}")
        return

    # --- Load APETokenizer vocab ---
    if args.vocab_file is not None:
        vocab_path = args.vocab_file
        print(f"Loading APETokenizer vocab from file: {vocab_path}")
    else:
        local_dir = os.path.join(os.path.dirname(__file__), 'downloaded')
        os.makedirs(local_dir, exist_ok=True)
        local_vocab_path = os.path.join(local_dir, 'tokenizer.json')
        if os.path.exists(local_vocab_path):
            print(f"Found existing vocab at: {local_vocab_path}")
            vocab_path = local_vocab_path
        else:
            print("No vocab file supplied, downloading from HuggingFace Hub...")
            vocab_path = hf_hub_download(
                repo_id="mikemayuare/SELFYAPE",
                filename="tokenizer.json",
            )
            shutil.copy(vocab_path, local_vocab_path)
            print(f"Downloaded vocab to: {vocab_path}")
            print(f"Copied vocab to local path: {local_vocab_path}")
            vocab_path = local_vocab_path
    tokenizer = APETokenizer()
    # Use wrapper to extract vocab if needed, then set directly
    vocab_dict = extract_hf_vocab(vocab_path)
    tokenizer.vocabulary = vocab_dict
    tokenizer.update_reverse_vocabulary()
    print("Vocabulary size:", len(tokenizer.vocabulary))

    # Set thread count if using CPU
    if args.device == 'cpu':
        torch.set_num_threads(126)

    def seed_everything(seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)  # must come before heavy imports
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)          # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    seed_everything(args.seed)

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
    if args.model_type == 'vaenovo':
        print("[INFO] Using VAENovo: minimal LSTM-based VAE for SMILES→SELFIES→tokens (APETokenizer)→reconstruction.")
        # Require vocab file
        if not args.vocab_file:
            print("[ERROR] --vocab_file must be provided for vaenovo with real APETokenizer vocab.")
            sys.exit(1)
        print(f"[INFO] Loading APETokenizer vocab from {args.vocab_file}")
        tokenizer = APETokenizer()
        tokenizer.load_vocabulary(args.vocab_file)
        print(f"[INFO] Loaded vocab size: {len(tokenizer)}")
        # Load SMILES from data files
        train_smiles = load_smiles_file('data_trn.txt')
        val_smiles = load_smiles_file('data_val.txt')
        test_smiles = load_smiles_file('data_tst.txt')
        print(f"[INFO] Loaded {len(train_smiles)} train, {len(val_smiles)} val, {len(test_smiles)} test molecules.")
        # For fast debug, use only first 1000 train/val/test
        N = 1000
        train_smiles = train_smiles[:N]
        val_smiles = val_smiles[:N]
        test_smiles = test_smiles[:N]
        vocab_size = len(tokenizer)
        device = torch.device(args.device)
        # model = VAENovo(vocab_size=vocab_size, emb_dim=1024, hidden_dim=256, latent_dim=128, num_layers=2, max_len=24).to(device)
        model = VAENovo(vocab_size=vocab_size, emb_dim=1024, hidden_dim=128*2, latent_dim=256*2, num_layers=2, max_len=24).to(device)
        # Prepare tokenized training set
        token_tensors = [model.string2tensor(s, tokenizer, device=device) for s in train_smiles]
        max_len = max(t.size(0) for t in token_tensors)
        padded = torch.full((len(token_tensors), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        for i, t in enumerate(token_tensors):
            padded[i, :t.size(0)] = t
        loader = torch.utils.data.DataLoader(padded, batch_size=16, shuffle=True)
        
        import datetime as dt
        
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.99, patience=6, min_lr=1e-7)
        tqdm_bar_args = dict(leave=True, ascii=True, ncols=100, dynamic_ncols=True)
        from misc import KLAnnealer
        kl_annealer = KLAnnealer(args.epochs, config)
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            kl_weight = kl_annealer(epoch)
            pbar = tqdm(loader, desc=f"Epoch {epoch}", **tqdm_bar_args)
            for batch_idx, batch in enumerate(pbar):
                recon_loss, kl_loss, total_loss = model(batch, kl_weight=kl_weight)
                opt.zero_grad()
                total_loss.backward()
                opt.step()
                epoch_loss += total_loss.item() * batch.size(0)
                epoch_recon += recon_loss.item() * batch.size(0)
                epoch_kl += kl_loss.item() * batch.size(0)
                avg_loss = epoch_loss / ((batch_idx+1) * batch.size(0))
                avg_recon = epoch_recon / ((batch_idx+1) * batch.size(0))
                avg_kl = epoch_kl / ((batch_idx+1) * batch.size(0))
                pbar.set_postfix(loss=f"{avg_loss:.4f}", recon=f"{avg_recon:.4f}", kl=f"{avg_kl:.4f}", klw=f"{kl_weight:.4f}")
                if args.debug or DEBUG:
                    print(f"[DEBUG] Epoch {epoch} Batch {batch_idx}: recon_loss={recon_loss.item():.4f}, kl_loss={kl_loss.item():.4f}, kl_weight={kl_weight:.4f}, total_loss={total_loss.item():.4f}")
                # Print reconstructions for first batch of each epoch
                if batch_idx == 0:
                    try:
                        decoded_in = [model.tensor2string(seq, tokenizer) for seq in batch[:5]]
                        z, _ = model.encode(batch[:5])
                        recon_logits = model.decode(z, batch[:5])
                        recon_tokens = torch.argmax(recon_logits, dim=-1)
                        decoded_out = [model.tensor2string(seq, tokenizer) for seq in recon_tokens]
                        print("[RECON] Input vs. Reconstruction (first 5):")
                        for i, (inp, outp) in enumerate(zip(decoded_in, decoded_out)):
                            print(f"  [{i}] {inp} => {outp}")
                    except Exception as e:
                        print(f"[DEBUG] Reconstruction print failed: {e}")
            epoch_loss /= len(loader.dataset)
            epoch_recon /= len(loader.dataset)
            epoch_kl /= len(loader.dataset)
            print(f"[EPOCH SUMMARY] Epoch {epoch}: avg_loss={epoch_loss:.4f}, avg_recon={epoch_recon:.4f}, avg_kl={epoch_kl:.4f}, kl_weight={kl_weight:.4f}")
            scheduler.step(epoch_loss)
            current_lr = opt.param_groups[0]['lr']
            print(f"[LR] Epoch {epoch+1}: lr={current_lr:.6g}")
            if epoch_loss < args.min_loss:
                print(f"[EARLY STOP] Stopping training at epoch {epoch+1} due to min_loss criterion: {epoch_loss:.6f} < {args.min_loss}")
                break
        # Evaluate on test set, save results
        model.eval()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = "test_results"
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"test_results_vaenovo_{timestamp}.txt")
        from rdkit import Chem
        
        with torch.no_grad(), open(out_path, 'w') as f:
            stats = collections.Counter()
            edit_distances = []
            can_edit_distances = []
            f.write("# VAENovo Test Results\n")
            for i, smi in enumerate(test_smiles):
                string2tensor_fn = model.module.string2tensor if hasattr(model, 'module') else model.string2tensor
                tensor = string2tensor_fn(smi, tokenizer, device=device)
                tensor = tensor.unsqueeze(0)
                recon_tokens = model.generate(tensor, device=device)
                recon_smiles = model.tensor2string(recon_tokens[0], tokenizer)
                input_tokens = tensor.squeeze(0).tolist()
                recon_tokens_list = recon_tokens[0].tolist() if hasattr(recon_tokens[0], 'tolist') else list(recon_tokens[0])
                # Canonicalize
                can_in = Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else ''
                can_out = Chem.MolToSmiles(Chem.MolFromSmiles(recon_smiles)) if Chem.MolFromSmiles(recon_smiles) else ''
                edit_dist = sum(a != b for a, b in zip(smi, recon_smiles)) + abs(len(smi) - len(recon_smiles))
                can_edit_dist = sum(a != b for a, b in zip(can_in, can_out)) + abs(len(can_in) - len(can_out))
                valid = (can_out != '')
                exact = (recon_smiles == smi)
                can_exact = (can_in == can_out) and (can_in != '')
                stats['total'] += 1
                if valid:
                    stats['valid'] += 1
                if exact:
                    stats['exact'] += 1
                if can_exact:
                    stats['can_exact'] += 1
                edit_distances.append(edit_dist)
                can_edit_distances.append(can_edit_dist)
                # Write as row-oriented block
                f.write(f"---\n")
                f.write(f"InputSMILES: {smi}\n")
                f.write(f"ReconSMILES: {recon_smiles}\n")
                f.write(f"InputTokens: {input_tokens}\n")
                f.write(f"ReconTokens: {recon_tokens_list}\n")
                f.write(f"EditDistance: {edit_dist}\n")
                f.write(f"CanInputSMILES: {can_in}\n")
                f.write(f"CanReconSMILES: {can_out}\n")
                f.write(f"CanEditDistance: {can_edit_dist}\n")
                f.write(f"Valid: {valid}\n")
                f.write(f"Exact: {exact}\n")
                f.write(f"CanExact: {can_exact}\n")
            # Compile and write statistics
            mean_ed = sum(edit_distances)/len(edit_distances) if edit_distances else 0.0
            mean_can_ed = sum(can_edit_distances)/len(can_edit_distances) if can_edit_distances else 0.0
            f.write("===\n")
            f.write(f"Total: {stats['total']}\n")
            f.write(f"Valid: {stats['valid']}\n")
            f.write(f"Exact: {stats['exact']}\n")
            f.write(f"CanExact: {stats['can_exact']}\n")
            f.write(f"MeanEditDistance: {mean_ed:.3f}\n")
            f.write(f"MeanCanEditDistance: {mean_can_ed:.3f}\n")
        print(f"Test results logged to {out_path}")
        print(f"Stats: {dict(stats)}")
        print(f"Avg edit distance: {sum(edit_distances)/len(edit_distances) if edit_distances else 0:.2f}")
        print(f"Avg canonical edit distance: {sum(can_edit_distances)/len(can_edit_distances) if can_edit_distances else 0:.2f}")
    else:
        print('Using VAEDummy (plain LSTM autoencoder)')
        model = VAEDummy(tokenizer, config)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_start)
    from misc import KLAnnealer
    kl_annealer = KLAnnealer(args.epochs, config)
    # , glob
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
   
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience, min_lr=config.lr_end)
    # Restore scheduler state if present
    if 'scheduler_state_dict' in metadata:
        scheduler.load_state_dict(metadata['scheduler_state_dict'])
    trainer = VAETrainer(config)
    trainer.fit(model, train_loader, val_loader, logger=logger, epochs=args.epochs, lr=config.lr_start, scheduler=scheduler, checkpoint_dir=checkpoint_dir, start_epoch=start_epoch, min_loss=args.min_loss)

    # Save final checkpoint after training
    print("[DEBUG] Training complete, about to call save_final_checkpoint")
    save_final_checkpoint(model, optimizer, end_epoch, best_val_loss, config)

    # save log with a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
                string2tensor_fn = model.module.string2tensor if hasattr(model, 'module') else model.string2tensor
                input_tensor = string2tensor_fn(s, device=args.device).unsqueeze(0)
                # Assert every token is a known token
                input_ids = input_tensor.squeeze(0).tolist()
                vocab_ids = set(range(len(tokenizer)))
                for token_id in input_ids:
                    assert token_id in vocab_ids, f"Unknown token_id {token_id} in input: {input_ids} (vocab size: {len(tokenizer)})"
                # Diagnostic: print the target (input) tokens and decoded SMILES for the first 5 samples
                if idx < 5:
                    token_strs = tokenizer.convert_ids_to_tokens(input_ids)
                    print(f"[TEST DIAG] Sample {idx}: input tokens = {input_ids}")
                    print(f"[TEST DIAG] Sample {idx}: token strings = {token_strs}")
                    # Remove special tokens for decoding
                    special_tokens = set([
                        tokenizer.pad_token,
                        tokenizer.bos_token,
                        tokenizer.eos_token,
                        tokenizer.unk_token,
                        tokenizer.mask_token
                    ])
                    filtered_tokens = [tok for tok in token_strs if tok not in special_tokens]
                    selfies_str = ''.join(filtered_tokens)
                    try:
                        decoded = selfies.decoder(selfies_str)
                    except Exception as e:
                        decoded = f"[DecoderError: {e}]"
                    print(f"[TEST DIAG] Sample {idx}: decoded SMILES = {decoded}")
                with torch.no_grad():
                    z, _ = model.forward_encoder(input_tensor)
                    # Get both decoded SMILES and output token tensor if possible
                    if hasattr(model, 'sample') and 'return_tensor' in model.sample.__code__.co_varnames:
                        out_decoded, out_token_tensors = model.sample(1, max_len=len(s)+10, z=z, return_tensor=True)
                        out = out_decoded[0]
                        out_token_tensor = out_token_tensors[0]
                        out_token_indices = out_token_tensor.tolist()
                        out_token_strs = tokenizer.convert_ids_to_tokens(out_token_indices)
                        if idx < 5:
                            print(f"[TEST DIAG] Sample {idx}: output token indices = {out_token_indices}")
                            print(f"[TEST DIAG] Sample {idx}: output token strings = {out_token_strs}")
                            # Remove special tokens for decoding
                            special_tokens = set([
                                tokenizer.pad_token,
                                tokenizer.bos_token,
                                tokenizer.eos_token,
                                tokenizer.unk_token,
                                tokenizer.mask_token
                            ])
                            filtered_tokens = [tok for tok in out_token_strs if tok not in special_tokens]
                            selfies_str = ''.join(filtered_tokens)
                            try:
                                decoded = selfies.decoder(selfies_str)
                            except Exception as e:
                                decoded = f"[DecoderError: {e}]"
                            print(f"[TEST DIAG] Sample {idx}: output decoded SMILES = {decoded}")
                    else:
                        out_tokens = model.sample(1, max_len=len(s)+10, z=z)
                        out = out_tokens[0]
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
                            decoded, out_tensors = model.sample(1, max_len=len(batch[i]), z=z, return_tensor=True)
                            out_tensor = out_tensors[0]
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
