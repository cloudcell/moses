import torch
from torch.nn.utils import clip_grad_norm_
from utils import Logger, CircularBuffer

class VAETrainer:
    def __init__(self, config):
        self.config = config

    def _train_epoch(self, model, epoch, train_loader, optimizer):
        model.train()
        kl_loss_values = CircularBuffer(self.config.n_last)
        recon_loss_values = CircularBuffer(self.config.n_last)
        loss_values = CircularBuffer(self.config.n_last)
        for input_batch in train_loader:
            input_batch = [data.to(model.device) for data in input_batch]
            kl_loss, recon_loss = model(input_batch)
            loss = kl_loss + recon_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), self.config.clip_grad)
            optimizer.step()
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())
        return {
            'epoch': epoch,
            'kl_loss': kl_loss_values.mean(),
            'recon_loss': recon_loss_values.mean(),
            'loss': loss_values.mean(),
        }

    def fit(self, model, train_loader, logger=None, epochs=10, lr=None):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr or self.config.lr_start)
        for epoch in range(epochs):
            stats = self._train_epoch(model, epoch, train_loader, optimizer)
            if logger is not None:
                logger.append(stats)
        return model
