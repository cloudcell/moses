import selfies
from apetokenizer.ape_tokenizer import APETokenizer
from data_loader import StringDataset
import json

# Load a few SMILES examples (edit as needed)
smiles_examples = [
    'CCO', 'c1ccccc1', 'CC(=O)O', 'O=C=O', 'C1CCCCC1', 'C2CCOCC2'
]

# Load tokenizer from vocab file (edit path as needed)
tokenizer_path = 'downloaded/tokenizer.json'
tokenizer = APETokenizer()
tokenizer.load_vocabulary(tokenizer_path)

print(f"Loaded vocab size: {len(tokenizer.vocabulary)}")

# Create dataset (this will print debug info)
dataset = StringDataset(smiles_examples, tokenizer=tokenizer)

# Show decoded output for first 5 token lists
to_decode = dataset.tokens[:5]
for i, toks in enumerate(to_decode):
    print(f'--- Example {i+1} ---')
    print('Tokens:', toks)
    decoded = dataset.decode_tokens(toks)
    print('Decoded:', decoded)
