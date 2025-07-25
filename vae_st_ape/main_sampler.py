import argparse
import torch
import selfies
from pathlib import Path

# Assume APETokenizer or similar is in load_tokenizer.py or standalone_tokenizer.py
# We'll try to import from both, fallback if needed
try:
    from load_tokenizer import APETokenizer
except ImportError:
    from standalone_tokenizer import APETokenizer

# Import VAEDummy2 directly
from model import VAEDummy2

def smiles_to_selfies(smiles: str) -> str:
    """Convert SMILES to SELFIES using selfies lib."""
    return selfies.encoder(smiles)

def main():
    parser = argparse.ArgumentParser(description="Sample from VAE model with hardcoded SMILES input.")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, help='Path to vocab file for tokenizer')
    parser.add_argument('--smiles', type=str, required=False, default='CCO', help='SMILES string to sample (default: ethanol)')
    args = parser.parse_args()

    # Step 1: Convert SMILES to SELFIES
    selfies_str = smiles_to_selfies(args.smiles)
    print(f"SMILES: {args.smiles}\nSELFIES: {selfies_str}")

    # Step 2: Tokenize SELFIES
    tokenizer = APETokenizer()
    tokenizer.load_vocabulary(args.vocab)
    token_ids = tokenizer.encode(selfies_str, add_special_tokens=True)
    print(f"Token IDs: {token_ids}")

    # Step 3: Prepare input tensor
    input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]

    # Step 4: Load model (VAEDummy2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    # Try to infer vocab size from tokenizer
    # Use config from checkpoint for model instantiation
    config = checkpoint.get('metadata', {}).get('config', {})
    vocab_size = len(tokenizer.vocabulary)
    emb_dim = config.get('emb_dim', 512)
    hidden_dim = config.get('hidden_dim', 256)
    num_layers_enc = config.get('num_layers_enc', 2)
    num_layers_dec = config.get('num_layers_dec', 2)
    max_len = config.get('max_len', 24)
    enc_dropout = config.get('enc_dropout', 0.1)
    dec_dropout = config.get('dec_dropout', 0.1)
    use_lstm = config.get('use_lstm', False)
    model = VAEDummy2(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        max_len=max_len,
        enc_dropout=enc_dropout,
        dec_dropout=dec_dropout,
        use_lstm=use_lstm
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        # (A) Reconstruction of input
        logits, z = model(input_tensor)
        recon_tokens = logits.argmax(-1).squeeze(0).tolist()
        recon_tokens_str = tokenizer.convert_ids_to_tokens(recon_tokens)
        recon_selfies = ''.join([t for t in recon_tokens_str if t not in tokenizer.special_tokens])
        try:
            recon_smiles = selfies.decoder(recon_selfies)
        except Exception as e:
            recon_smiles = f"[SELFIES decode error: {e}]"
        print(f"Reconstruction SELFIES: {recon_selfies}")
        print(f"Reconstruction SMILES: {recon_smiles}")

        # Canonicalize input and reconstructed SMILES
        try:
            from rdkit import Chem
            # Canonical input
            mol_in = Chem.MolFromSmiles(args.smiles)
            can_in = Chem.MolToSmiles(mol_in, canonical=True) if mol_in is not None else '[Invalid input SMILES]'
            print(f"Canonical Input SMILES: {can_in}")
            # Canonical reconstruction
            mol_rec = Chem.MolFromSmiles(recon_smiles)
            can_rec = Chem.MolToSmiles(mol_rec, canonical=True) if mol_rec is not None else '[Invalid reconstructed SMILES]'
            print(f"Canonical Reconstructed SMILES: {can_rec}")
        except ImportError:
            print("[RDKit not installed: cannot canonicalize SMILES]")
        
        # compare can_in and can_rec
        if can_in == can_rec:
            print("Input and reconstructed SMILES are the same")
        else:
            print("Input and reconstructed SMILES are different")

        # (B) Unconditional sample from prior
        sample_tokens = model.sample(batch_size=1).squeeze(0).tolist()
        sample_tokens_str = tokenizer.convert_ids_to_tokens(sample_tokens)
        sample_selfies = ''.join([t for t in sample_tokens_str if t not in tokenizer.special_tokens])
        try:
            sample_smiles = selfies.decoder(sample_selfies)
        except Exception as e:
            sample_smiles = f"[SELFIES decode error: {e}]"
        print(f"Sampled SELFIES: {sample_selfies}")
        print(f"Sampled SMILES: {sample_smiles}")

if __name__ == "__main__":
    main()
