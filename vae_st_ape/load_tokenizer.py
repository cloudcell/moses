"""
Script to download and load APETokenizer vocabulary from HuggingFace Hub.
Usage:
    python load_tokenizer.py
"""
from huggingface_hub import hf_hub_download
from apetokenizer.ape_tokenizer import APETokenizer

# ① Download the vocab (no model weights, no config – one file only)
vocab_path = hf_hub_download(
    repo_id="mikemayuare/SELFYAPE",  # any repo that carries a compatible APE vocab
    filename="tokenizer.json",             # <- file name shown in the model card
)

# ② Initialise your tokenizer with that vocab
tok = APETokenizer()
tok.load_vocabulary(vocab_path)            # your normal helper from the repo

print("Vocabulary size:", len(tok.vocabulary))
print("Vocab loaded from:", vocab_path)

# Optionally: save for later use
# tok.save_vocabulary("./ape_tokenizer_vocab.json")
