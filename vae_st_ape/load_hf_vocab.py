import json

def extract_hf_vocab(vocab_path):
    """
    Loads a HuggingFace-style tokenizer.json and extracts the ['model']['vocab'] dict if present.
    Falls back to loading the file as a flat vocab dict if not.
    Returns a {token: id} dict.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
        if isinstance(obj, dict) and 'model' in obj and 'vocab' in obj['model']:
            return obj['model']['vocab']
        return obj
