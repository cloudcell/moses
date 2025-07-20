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
        from tqdm import tqdm
        train_iter = tqdm(train_loader, desc=f"Train (epoch {epoch+1})")
        for input_batch in train_iter:
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
            train_iter.set_postfix({
                # as 6 digits after the decimal point
                'loss': f"{loss.item():.6f}", 
                'kl': f"{kl_loss.item():.6f}", 
                'recon': f"{recon_loss.item():.6f}", 
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}", 
            })
        return {
            'epoch': epoch + 1,
            'kl_loss': kl_loss_values.mean(),
            'recon_loss': recon_loss_values.mean(),
            'loss': loss_values.mean(),
        }

    def fit(self, model, train_loader, val_loader, logger=None, epochs=10, lr=None, scheduler=None, checkpoint_dir=None, start_epoch=0):
        import os
        if scheduler is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr or self.config.lr_start)
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.config.lr_factor, patience=self.config.lr_patience, min_lr=self.config.lr_end)
        else:
            optimizer = scheduler.optimizer
        for epoch in range(start_epoch, start_epoch + epochs):
            stats = self._train_epoch(model, epoch, train_loader, optimizer)
            # After each epoch, evaluate on validation set to get val_loss
            model.eval()
            val_loss = 0.0
            val_batches = 0
            from tqdm import tqdm
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Val (epoch {epoch+1})")
                for batch in val_iter:
                    batch = [b.to(model.device) for b in batch]
                    kl_loss, recon_loss = model(batch)
                    loss = kl_loss + recon_loss
                    val_loss += loss.item()
                    val_batches += 1
                    val_iter.set_postfix({
                        # as 6 digits after the decimal point
                        'loss': f"{loss.item():.6f}", 
                        'kl': f"{kl_loss.item():.6f}", 
                        'recon': f"{recon_loss.item():.6f}", 
                        'lr': f"{optimizer.param_groups[0]['lr']:.6f}", 
                    })
            mean_val_loss = val_loss / max(1, val_batches)
            scheduler.step(mean_val_loss)
            if logger is not None:
                stats['val_loss'] = mean_val_loss
                logger.append(stats)
            # Save checkpoint with full metadata
            if checkpoint_dir is not None:
                epoch_str = f"{epoch+1:05d}"
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_str}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metadata': {
                        'epoch': epoch+1,
                        'val_loss': mean_val_loss,
                        'config': self.config.__dict__,
                        'scheduler_state_dict': scheduler.state_dict(),
                    }
                }, checkpoint_path)
        return model
