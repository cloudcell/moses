class VAEConfig:
    """
    Configuration for VAE model. Add 'max_len' for maximum sequence length in tokens.
    """
    def __init__(self):
        # Model params
        
        self.q_cell = 'gru'
        self.q_dropout = 0.0
        self.q_bidir = True
        self.q_d_h = 256 
        self.q_n_layers = 3 # 2 worked

        self.d_cell = 'gru'
        self.d_dropout = 0.0
        self.d_n_layers = 3
        self.d_d_h = 512
        self.d_z = 128 * 2


        self.freeze_embeddings = False
        # Maximum sequence length (in tokens)
        self.max_len = 24 #60  # Default maximum number of tokens per sequence
        # Train params
        self.n_batch = 512
        self.clip_grad = 50
        self.kl_start = 0
        self.kl_w_start = 0.50
        self.kl_w_end = 0.05  # 0.05
        self.lr_start = 1e-3  #3e-3
        self.lr_end = 1e-7  #3e-4
        self.lr_factor = 0.95
        self.lr_patience = 3
        self.lr_n_period = 10
        self.lr_n_restarts = 10
        self.lr_n_mult = 1
        self.n_last = 1000
        self.n_jobs = 1
        self.n_workers = 1
        self.log_file = None
        self.model_save = None
        self.save_frequency = 1

def get_default_config():
    return VAEConfig()


#   emb_dim=cfg.emb_dim, # 512*2, 
#   hidden_dim=cfg.hidden_dim, # 128 *2, 
#   num_layers=cfg.num_layers, # 1, 
#   max_len=cfg.max_len, # 24, 
#   enc_dropout=cfg.enc_dropout, # 0.1, 
#   dec_dropout=cfg.dec_dropout # 0.1


class VAEDummyConfig:
    def __init__(self):
        self.max_len = 24
        self.emb_dim = 512*2
        self.hidden_dim = 128 *2
        self.num_layers = 1
        self.enc_dropout = 0.01 if self.num_layers > 1 else 0.0
        self.dec_dropout = 0.01 if self.num_layers > 1 else 0.0
        self.lr_start = 1e-3
        self.lr_end = 1e-8
        self.lr_factor = 0.95
        self.lr_patience = 3
        
        
def get_vaedummy_config():
    return VAEDummyConfig()