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
        self.selfies = []
        self.skipped = 0
        for s in self.smiles:
            try:
                sf = selfies.encoder(s)
                self.selfies.append(sf)
            except selfies.exceptions.EncoderError as e:
                print(f"[WARN] Skipping SMILES not convertible to SELFIES: {s}\nReason: {e}")
                self.skipped += 1
        if self.skipped > 0:
            print(f"[INFO] Skipped {self.skipped} SMILES in dataset due to SELFIES encoding errors.")
        print(f"[SUMMARY] Total SMILES: {len(self.smiles)} | Encoded SELFIES: {len(self.selfies)} | Skipped: {self.skipped}")
        if len(self.selfies) == 0:
            print("[FATAL] All SMILES were skipped. Check your data file and format.")
        # Require tokenizer to be provided and loaded externally
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided and loaded externally (e.g., from a vocab file). Do not train inside data loader.")
        self.tokenizer = tokenizer
        # Check for <unk> tokens in first 5 tokenized entries
        tokens = [self.tokenizer.encode(sf, add_special_tokens=True) for sf in self.selfies]
        unk_id = self.tokenizer.vocabulary.get(self.tokenizer.unk_token, -999)
        with open('token_debug.log', 'w') as dbgfile:
            for i, tlist in enumerate(tokens[:5]):
                n_unk = sum(1 for tid in tlist if tid == unk_id)
                dbg_line = f"[DEBUG] Entry {i}: {n_unk} <unk> tokens in {tlist}"
                print(dbg_line)
                dbgfile.write(dbg_line + '\n')
        # Require tokenizer to be provided and loaded externally
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided and loaded externally (e.g., from a vocab file). Do not train inside data loader.")
        self.tokenizer = tokenizer
        # Tokenize all selfies
        max_len = getattr(config, 'max_len', None)
        # Debug: Show first 5 SMILES and their SELFIES
        print("[DEBUG] First 5 SMILES:", self.smiles[:5])
        print("[DEBUG] First 5 SELFIES:", self.selfies[:5])
        tokens = [self.tokenizer.encode(sf, add_special_tokens=True) for sf in self.selfies]
        print("[DEBUG] First 5 tokenized SELFIES:", tokens[:5])
        if max_len is not None:
            self.tokens = [t[:max_len] + [self.tokenizer.pad_token_id]*(max_len-len(t)) if len(t)<max_len else t[:max_len] for t in tokens]
        else:
            self.tokens = tokens
    def __len__(self):
        return len(self.selfies)
    def __getitem__(self, idx):
        return self.tokens[idx]
    def decode_tokens(self, tokens):
        # tokens: list of ints
        token_strs = self.tokenizer.convert_ids_to_tokens(tokens)
        # Filter out special tokens
        special_tokens = set([
            self.tokenizer.pad_token,
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.unk_token,
            self.tokenizer.mask_token
        ])
        filtered_tokens = [tok for tok in token_strs if tok not in special_tokens]
        print(f"[DEBUG] decode_tokens: tokens={tokens}\ntoken_strs={token_strs}\nfiltered_tokens={filtered_tokens}")
        selfies_str = ''.join(filtered_tokens)
        try:
            decoded = selfies.decoder(selfies_str)
        except Exception as e:
            print(f"[ERROR] SELFIES decoding failed for: {selfies_str}\nError: {e}")
            decoded = ''
        return decoded
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
