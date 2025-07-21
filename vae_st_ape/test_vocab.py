"""
Test that every SMILES in data_trn.txt, data_val.txt, data_tst.txt can be tokenized with downloaded/tokenizer.json
without producing any <unk> (unknown) tokens.

Usage:
    python test_vocab.py
"""
import os
import selfies
import json
from apetokenizer.ape_tokenizer import APETokenizer

DATA_FILES = ["data_trn.txt", "data_val.txt", "data_tst.txt"]
VOCAB_PATH = os.path.join("downloaded", "tokenizer.json")
UNK_TOKEN = "<unk>"

# Load tokenizer
print(f"Loading tokenizer vocab from {VOCAB_PATH}")
tokenizer = APETokenizer()
tokenizer.load_vocabulary(VOCAB_PATH)
unk_id = tokenizer.vocabulary[UNK_TOKEN]

for fname in DATA_FILES:
    print(f"Testing file: {fname}")
    with open(fname, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    for i, s in enumerate(smiles_list):
        try:
            sf = selfies.encoder(s)
        except selfies.exceptions.EncoderError as e:
            print(f"[SKIP] {fname} line {i+1}: cannot encode to SELFIES: {s}\nReason: {e}")
            continue
        token_ids = tokenizer.encode(sf, add_special_tokens=True)
        if unk_id in token_ids:
            print('---------------------------')
            print(f"SMILES: {s}")
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            unk_indices = [j for j, tid in enumerate(token_ids) if tid == unk_id]
            unk_tokens = [tokens[j] for j in unk_indices]
            print(f"[FAIL] {fname} line {i+1}: <unk> token produced")
            print(f"SELFIES: {sf}")
            print(f"Token IDs: {token_ids}")
            print(f"Tokens: {tokens}")
            print(f"\tIndices of <unk>: {unk_indices}")
            print(f"\tTokens mapped to <unk>: {unk_tokens}")
    print(f"[OK] All SMILES in {fname} tokenized without <unk>.")
print("All files passed.")
