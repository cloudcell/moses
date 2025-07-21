import torch
from torch.utils.data import Dataset, DataLoader
import selfies
from apetokenizer.ape_tokenizer import APETokenizer

# Optionally: path to pretrained tokenizer vocabulary
TOKENIZER_VOCAB_PATH = None  # Set to string path if needed

class StringDataset(Dataset):
    """
    Dataset for SMILES/SELFIES strings. Uses config.max_len for truncation/padding if provided.
    """
    def __init__(self, smiles_list, tokenizer=None, config=None):
        self.smiles = smiles_list
        self.selfies = [selfies.encoder(s) for s in self.smiles]
        # Require tokenizer to be provided and loaded externally
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided and loaded externally (e.g., from a vocab file). Do not train inside data loader.")
        self.tokenizer = tokenizer
        # Tokenize all selfies
        max_len = getattr(config, 'max_len', None)
        tokens = [self.tokenizer.encode(sf, add_special_tokens=True) for sf in self.selfies]
        if max_len is not None:
            self.tokens = [t[:max_len] + [self.tokenizer.pad_token_id]*(max_len-len(t)) if len(t)<max_len else t[:max_len] for t in tokens]
        else:
            self.tokens = tokens
    def __len__(self):
        return len(self.smiles)
    def __getitem__(self, idx):
        return self.tokens[idx]
    def decode_tokens(self, tokens):
        # tokens: list of ints
        selfies_str = ''.join(self.tokenizer.convert_ids_to_tokens(tokens))
        return selfies.decoder(selfies_str)
# Note: Tokenizer must be loaded externally and passed to get_data_loaders.
def get_data_loaders(smiles_list, batch_size=64, shuffle=True, tokenizer=None, config=None):
    dataset = StringDataset(smiles_list, tokenizer=tokenizer, config=config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: [torch.tensor(seq, dtype=torch.long) for seq in batch]
    )
    return loader, dataset.tokenizer
