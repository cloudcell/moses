class VAEConfig:
    def __init__(self):
        # Model params
        self.q_cell = 'gru'
        self.q_bidir = False
        self.q_d_h = 256
        self.q_n_layers = 1
        self.q_dropout = 0.5
        self.d_cell = 'gru'
        self.d_n_layers = 3
        self.d_dropout = 0
        self.d_z = 128
        self.d_d_h = 512
        self.freeze_embeddings = False
        # Train params
        self.n_batch = 512
        self.clip_grad = 50
        self.kl_start = 0
        self.kl_w_start = 0
        self.kl_w_end = 0.05
        self.lr_start = 1e-4  #3e-3
        self.lr_factor = 0.9
        self.lr_patience = 0
        self.lr_n_period = 10
        self.lr_n_restarts = 10
        self.lr_n_mult = 1
        self.lr_end = 1e-6  #3e-4
        self.n_last = 1000
        self.n_jobs = 1
        self.n_workers = 1
        self.log_file = None
        self.model_save = None
        self.save_frequency = 1

def get_default_config():
    return VAEConfig()
