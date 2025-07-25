"""
Standalone script to train a tokenizer from all data_*.txt files in the current folder.
Outputs tokenizer_<timestamp>.json to ./tokenizers.

Usage:
    python standalone_tokenizer.py
"""
import os
import glob
import selfies
import datetime
from apetokenizer.ape_tokenizer import APETokenizer

# 1. Find all data_*.txt files in CWD
DATA_FILES = sorted(glob.glob("data_*.txt"))
if not DATA_FILES:
    print("No data_*.txt files found in current directory.")
    exit(1)
print(f"Found {len(DATA_FILES)} data files: {DATA_FILES}")

# 2. Read all SMILES
all_smiles = []
for fname in DATA_FILES:
    with open(fname, "r") as f:
        smiles = [line.strip() for line in f if line.strip()]
        all_smiles.extend(smiles)
print(f"Total SMILES loaded: {len(all_smiles)}")

# only unique SMILES
all_smiles = list(set(all_smiles))
print(f"Total unique SMILES: {len(all_smiles)}")


# 3. Convert to SELFIES, skipping unconvertible ones
all_selfies = []
skipped = 0
for s in all_smiles:
    try:
        sf = selfies.encoder(s)
        all_selfies.append(sf)
    except selfies.exceptions.EncoderError as e:
        print(f"[WARN] Skipping SMILES not convertible to SELFIES: {s}\nReason: {e}")
        skipped += 1
print(f"Total skipped SMILES: {skipped}")



# 4. Train tokenizer
print(f"Training tokenizer on {len(all_selfies)} molecules (SELFIES)...")
tokenizer = APETokenizer()
tokenizer.train(all_selfies, type="selfies", max_vocab_size=64)

# 5. Output to ./tokenizers/tokenizer_<timestamp>.json
os.makedirs("tokenizers", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join("tokenizers", f"tokenizer_{timestamp}.json")
tokenizer.save_vocabulary(out_path)
print(f"Tokenizer saved to {out_path}")
