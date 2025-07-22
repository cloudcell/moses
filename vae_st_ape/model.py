import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from main import DEBUG
except ImportError:
    DEBUG = False

class VAEDummy2(nn.Module):
    """
    LSTM encoder-decoder for sequence experiments. Supports SMILES → SELFIES → tokens (APETokenizer) and token ids → SMILES.
    Use for debugging or as a minimal molecular autoencoder.
    """
    def __init__(self, vocab_size=10, emb_dim=16, hidden_dim=32, num_layers=1, max_len=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, smiles, tokenizer, device=None):
        """
        Convert SMILES to SELFIES, then to token ids using APETokenizer.
        Returns a tensor of token ids.
        """
        import selfies
        selfies_str = selfies.encoder(smiles)
        ids = tokenizer.encode(selfies_str, add_special_tokens=True)
        device = device or self.device
        return torch.tensor(ids, dtype=torch.long, device=device)

    def tensor2string(self, token_ids, tokenizer):
        """
        Convert token ids to tokens, join to SELFIES, decode to SMILES. Prints all intermediate steps for debugging.
        """
        import selfies
        ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)
        print(f"[tensor2string] token_ids: {ids}")
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"[tensor2string] tokens: {tokens}")
        # Remove special tokens
        if hasattr(tokenizer, 'special_tokens') and tokenizer.special_tokens:
            special_tokens = set(tokenizer.special_tokens.keys())
        else:
            special_tokens = {'<unk>', '<pad>', '<s>', '</s>', '<mask>'}
        filtered_tokens = [tok for tok in tokens if tok not in special_tokens]
        print(f"[tensor2string] filtered tokens: {filtered_tokens}")
        selfies_str = ''.join(filtered_tokens)
        print(f"[tensor2string] selfies_str: '{selfies_str}'")
        if not filtered_tokens:
            print("[tensor2string][WARNING] All tokens filtered out as special tokens. Trying with unfiltered tokens.")
            selfies_str = ''.join(tokens)
            print(f"[tensor2string] selfies_str (unfiltered): '{selfies_str}'")
        try:
            smiles = selfies.decoder(selfies_str)
        except Exception as e:
            print(f"[Warning] Failed to decode SELFIES: '{selfies_str}'. Error: {e}")
            smiles = ''
        print(f"[tensor2string] decoded SMILES: '{smiles}'")
        return smiles

    def forward(self, x):
        # x: [batch, seq_len] integer tokens
        emb = self.embedding(x)
        _, (h, c) = self.encoder(emb)
        # Decoder input: shift x right, prepend 0 (start token)
        dec_in = torch.zeros_like(x)
        dec_in[:, 1:] = x[:, :-1]
        dec_emb = self.embedding(dec_in)
        out, _ = self.decoder(dec_emb, (h, c))
        logits = self.fc_out(out)
        return logits

    def sample(self, batch_size=1, max_len=None, device=None, h=None, c=None):
        if max_len is None:
            max_len = self.max_len
        if device is None:
            device = self.device
        # Start with start token 0
        inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        if h is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        if c is None:
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        outputs = []
        for _ in range(max_len):
            emb = self.embedding(inputs[:, -1:])
            out, (h, c) = self.decoder(emb, (h, c))
            logits = self.fc_out(out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            outputs.append(next_token)
            inputs = torch.cat([inputs, next_token], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs


class VAEDummy(nn.Module):
    def forward_encoder(self, x):
        # For compatibility: return encoder hidden state and kl_loss=0
        lengths = [len(i_x) for i_x in x]
        x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.pad)
        x_emb = self.x_emb(x_padded)
        packed = nn.utils.rnn.pack_padded_sequence(x_emb, lengths, batch_first=True, enforce_sorted=False)
        _, (h, c) = self.encoder_rnn(packed)
        if self.encoder_rnn.bidirectional:
            num_layers = self.encoder_rnn.num_layers
            h_fwd = h[0::2]
            h_bwd = h[1::2]
            h_dec = h_fwd + h_bwd
        else:
            h_dec = h
        # Use the last layer's hidden state as a dummy "z"
        z = h_dec[-1]
        kl_loss = torch.tensor(0.0, device=self.device)
        return z, kl_loss

    def __init__(self, tokenizer, config):
        super().__init__()
        self.config = config
        self.vocabulary = tokenizer
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.unk = tokenizer.special_tokens[tokenizer.unk_token] if hasattr(tokenizer, 'special_tokens') else tokenizer.unk_token_id
        n_vocab = len(tokenizer)
        d_emb = getattr(config, 'embedding_dim', 256)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        if getattr(config, 'freeze_embeddings', False):
            self.x_emb.weight.requires_grad = False
        self.encoder_rnn = nn.LSTM(
            d_emb,
            config.q_d_h,
            num_layers=config.q_n_layers,
            batch_first=True,
            dropout=config.q_dropout if config.q_n_layers > 1 else 0,
            bidirectional=config.q_bidir
        )
        self.decoder_rnn = nn.LSTM(
            d_emb,
            config.d_d_h,
            num_layers=config.d_n_layers,
            batch_first=True,
            dropout=config.d_dropout if config.d_n_layers > 1 else 0
        )
        # Add linear layers to project encoder hidden/cell state to decoder hidden size
        self.enc2dec_h = nn.Linear(config.q_d_h, config.d_d_h)
        self.enc2dec_c = nn.Linear(config.q_d_h, config.d_d_h)
        self.decoder_fc = nn.Linear(config.d_d_h, n_vocab)
        assert self.decoder_fc.out_features == n_vocab
        assert self.x_emb.num_embeddings == n_vocab
    @property
    def device(self):
        return next(self.parameters()).device
    def string2tensor(self, smiles, device='model'):
        import selfies
        selfies_str = selfies.encoder(smiles)
        ids = self.vocabulary.encode(selfies_str, add_special_tokens=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )
        return tensor
    def tensor2string(self, tensor):
        import selfies
        ids = tensor.tolist()
        tokens = self.vocabulary.convert_ids_to_tokens(ids)
        special_tokens = set()
        if hasattr(self.vocabulary, 'special_tokens'):
            special_tokens = set(self.vocabulary.special_tokens.keys())
        else:
            special_tokens = {'<unk>', '<pad>', '<s>', '</s>', '<mask>'}
        tokens = [tok for tok in tokens if tok not in special_tokens]
        selfies_str = ''.join(tokens)
        try:
            smiles = selfies.decoder(selfies_str)
        except Exception as e:
            print(f"[Warning] Failed to decode SELFIES: '{selfies_str}'. Error: {e}")
            smiles = ''
        return smiles
    def forward(self, x):
        # x: list of token tensors (batch)
        lengths = [len(i_x) for i_x in x]
        x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.pad)
        x_emb = self.x_emb(x_padded)
        packed = nn.utils.rnn.pack_padded_sequence(x_emb, lengths, batch_first=True, enforce_sorted=False)
        _, (h, c) = self.encoder_rnn(packed)
        # Combine bidirectional states for decoder init
        if self.encoder_rnn.bidirectional:
            num_layers = self.encoder_rnn.num_layers
            # h, c: [num_layers * 2, batch, hidden_size]
            h_fwd = h[0::2]
            h_bwd = h[1::2]
            c_fwd = c[0::2]
            c_bwd = c[1::2]
            h_dec = h_fwd + h_bwd
            c_dec = c_fwd + c_bwd
        else:
            h_dec = h
            c_dec = c
        # Project encoder hidden/cell state to decoder hidden size
        # h_dec, c_dec: [num_layers, batch, encoder_hidden_size]
        h_dec = self.enc2dec_h(h_dec)
        c_dec = self.enc2dec_c(c_dec)
        # Decoder: teacher forcing
        packed_dec = nn.utils.rnn.pack_padded_sequence(x_emb, lengths, batch_first=True, enforce_sorted=False)
        if DEBUG: print(f"[DEBUG] x_emb.shape: {x_emb.shape}")
        if DEBUG: print(f"[DEBUG] packed_dec.data.shape: {packed_dec.data.shape}")
        if DEBUG: print(f"[DEBUG] h_dec.shape: {h_dec.shape}")
        if DEBUG: print(f"[DEBUG] c_dec.shape: {c_dec.shape}")
        output, _ = self.decoder_rnn(packed_dec, (h_dec, c_dec))
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if DEBUG: print(f"[DEBUG] output.shape: {output.shape}")
        y = self.decoder_fc(output)
        if DEBUG: print(f"[DEBUG] y.shape: {y.shape}")
        if DEBUG: print(f"[DEBUG] x_padded.shape: {x_padded.shape}")
        if DEBUG: print(f"[DEBUG] y[:, :-1].shape: {y[:, :-1].shape}")
        if DEBUG: print(f"[DEBUG] x_padded[:, 1:].shape: {x_padded[:, 1:].shape}")
        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x_padded[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )
        kl_loss = torch.tensor(0.0, device=self.device)
        return kl_loss, recon_loss
    def sample(self, n_batch, max_len=None, z=None, temp=1.0, return_tensor=False):
        if max_len is None:
            max_len = getattr(self, 'config', None) and getattr(self.config, 'max_len', 100) or 100
        with torch.no_grad():
            # If z is not provided, encode random input (or use zeros)
            if z is None:
                z = torch.randn(n_batch, self.config.q_d_h, device=self.device)
            # Project z to decoder hidden/cell state using enc2dec_h/c
            h_proj = self.enc2dec_h(z)
            c_proj = self.enc2dec_c(z)
            if h_proj.dim() == 2:
                h_0 = h_proj.unsqueeze(0).repeat(self.config.d_n_layers, 1, 1)
                c_0 = c_proj.unsqueeze(0).repeat(self.config.d_n_layers, 1, 1)
            elif h_proj.dim() == 3:
                h_0 = h_proj
                c_0 = c_proj
            else:
                raise ValueError(f"Unexpected z shape after projection: {h_proj.shape}")
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch, max_len)
            x[:, 0] = self.bos
            eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=self.device)
            end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
            h = (h_0, c_0)
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                o, h = self.decoder_rnn(x_emb, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)
                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = (~eos_mask) & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])
            decoded = [self.tensor2string(i_x) for i_x in new_x]
            if return_tensor:
                return decoded, [i_x.cpu() for i_x in new_x]
            return decoded

class VAE(nn.Module):
    def __init__(self, tokenizer, config):
        self.config = config
        super().__init__()
        self.vocabulary = tokenizer
        # Use APETokenizer's special token ids
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.unk = tokenizer.special_tokens[tokenizer.unk_token] if hasattr(tokenizer, 'special_tokens') else tokenizer.unk_token_id
        n_vocab = len(tokenizer)
        d_emb = getattr(config, 'embedding_dim', 256)  # fallback to 256 if not set in config
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        # No pretrained vectors: random init
        if getattr(config, 'freeze_embeddings', False):
            self.x_emb.weight.requires_grad = False
        # the constructor doesn’t need to know the 
        # sequence length beforehand—it remains dynamic, 
        # defined by the data given to the GRU in the forward pass.
        if config.q_cell == 'gru':
            self.encoder_rnn = nn.GRU(
                d_emb,
                config.q_d_h,
                num_layers=config.q_n_layers,
                batch_first=True,
                dropout=config.q_dropout if config.q_n_layers > 1 else 0,
                bidirectional=config.q_bidir
            )
        else:
            raise ValueError("Invalid q_cell type, should be one of the ('gru',)")
        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, config.d_z)
        self.q_logvar = nn.Linear(q_d_last, config.d_z)
        if config.d_cell == 'gru':
            self.decoder_rnn = nn.GRU(
                d_emb + config.d_z,
                config.d_d_h,
                num_layers=config.d_n_layers,
                batch_first=True,
                dropout=config.d_dropout if config.d_n_layers > 1 else 0
            )
        else:
            raise ValueError("Invalid d_cell type, should be one of the ('gru',)")
        self.decoder_lat = nn.Linear(config.d_z, config.d_d_h)
        self.decoder_fc = nn.Linear(config.d_d_h, n_vocab)
        # Assert output and embedding match vocab size
        assert self.decoder_fc.out_features == n_vocab, f"decoder_fc out_features ({self.decoder_fc.out_features}) != vocab size ({n_vocab})"
        assert self.x_emb.num_embeddings == n_vocab, f"embedding num_embeddings ({self.x_emb.num_embeddings}) != vocab size ({n_vocab})"
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_lat,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])
    @property
    def device(self):
        return next(self.parameters()).device
    def string2tensor(self, smiles, device='model'):
        import selfies
        # Convert SMILES to SELFIES
        selfies_str = selfies.encoder(smiles)
        # Tokenize SELFIES using APETokenizer
        ids = self.vocabulary.encode(selfies_str, add_special_tokens=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )
        return tensor

    def tensor2string(self, tensor):
        import selfies
        ids = tensor.tolist()
        tokens = self.vocabulary.convert_ids_to_tokens(ids)
        # Remove all special tokens (not just <unk>)
        special_tokens = set()
        if hasattr(self.vocabulary, 'special_tokens'):
            special_tokens = set(self.vocabulary.special_tokens.keys())
        else:
            # Fallback: common special tokens
            special_tokens = {'<unk>', '<pad>', '<s>', '</s>', '<mask>'}
        tokens = [tok for tok in tokens if tok not in special_tokens]
        selfies_str = ''.join(tokens)
        try:
            smiles = selfies.decoder(selfies_str)
        except Exception as e:
            print(f"[Warning] Failed to decode SELFIES: '{selfies_str}'. Error: {e}")
            smiles = ''
        return smiles

    def forward(self, x):
        z, kl_loss = self.forward_encoder(x)
        recon_loss = self.forward_decoder(x, z)
        return kl_loss, recon_loss
    def forward_encoder(self, x):
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        _, h = self.encoder_rnn(x, None)
        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        return z, kl_loss
    def forward_decoder(self, x, z):
        lengths = [len(i_x) for i_x in x]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.pad)
        x_emb = self.x_emb(x)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True, enforce_sorted=False)
        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        output, _ = self.decoder_rnn(x_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)
        if DEBUG: print(f"[DEBUG] output.shape: {output.shape}")
        if DEBUG: print(f"[DEBUG] y.shape: {y.shape}")
        if DEBUG: print(f"[DEBUG] x.shape: {x.shape}")
        if DEBUG: print(f"[DEBUG] y[:, :-1].shape: {y[:, :-1].shape}")
        if DEBUG: print(f"[DEBUG] x[:, 1:].shape: {x[:, 1:].shape}")
        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )
        return recon_loss
    def sample_z_prior(self, n_batch):
        return torch.randn(n_batch, self.q_mu.out_features, device=self.x_emb.weight.device)
    def sample(self, n_batch, max_len=None, z=None, temp=1.0):
        if max_len is None:
            max_len = getattr(self, 'config', None) and getattr(self.config, 'max_len', 100) or 100
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch, max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=self.device)
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)
                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)
                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = (~eos_mask) & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])
            # Return list of decoded SMILES, marking any decoding failure as ''
            return [self.tensor2string(i_x) for i_x in new_x]
