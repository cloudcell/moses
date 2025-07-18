import argparse
import os

# This script extracts the first column (SMILES) from a .smi file and saves as a text file (one SMILES per line)
import random

def extract_and_split_smiles(input_path, out_prefix, limit=None, seed=42):
    smiles = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            first_col = line.split()[0]
            smiles.append(first_col)
    print(f"Extracted {len(smiles)} SMILES from {input_path}")
    # Shuffle and split
    random.seed(seed)
    random.shuffle(smiles)
    n = len(smiles)
    n_trn = int(0.8 * n)
    n_val = int(0.1 * n)
    n_tst = n - n_trn - n_val
    trn = smiles[:n_trn]
    val = smiles[n_trn:n_trn+n_val]
    tst = smiles[n_trn+n_val:]
    with open(f"{out_prefix}_trn.txt", 'w') as f:
        for s in trn:
            f.write(s + '\n')
    with open(f"{out_prefix}_val.txt", 'w') as f:
        for s in val:
            f.write(s + '\n')
    with open(f"{out_prefix}_tst.txt", 'w') as f:
        for s in tst:
            f.write(s + '\n')
    print(f"Saved {len(trn)} train, {len(val)} val, {len(tst)} test SMILES to {out_prefix}_trn.txt, {out_prefix}_val.txt, {out_prefix}_tst.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data/11_p0.smi', help='Input .smi file')
    parser.add_argument('--prefix', type=str, default='data', help='Output file prefix (will create _trn.txt, _val.txt, _tst.txt)')
    parser.add_argument('--limit', type=int, default=None, help='Optional: limit number of SMILES')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    args = parser.parse_args()
    extract_and_split_smiles(args.input, args.prefix, limit=args.limit, seed=args.seed)
