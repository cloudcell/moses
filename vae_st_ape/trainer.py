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
        total_loss = 0.0
        total_kl = 0.0
        total_recon = 0.0
        total_samples = 0
        from tqdm import tqdm
        train_iter = tqdm(train_loader, desc=f"Train (epoch {epoch+1})")
        early_stop = False
        for input_batch in train_iter:
            input_batch = [data.to(model.device) for data in input_batch]
            batch_size = input_batch[0].size(0) if hasattr(input_batch[0], 'size') else len(input_batch[0])
            kl_loss, recon_loss = model(input_batch)
            loss = kl_loss + recon_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), self.config.clip_grad)
            optimizer.step()
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())
            total_loss += loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_samples += batch_size
            train_iter.set_postfix({
                'loss': f"{total_loss / max(1, total_samples):.6f}",
                'kl': f"{total_kl / max(1, total_samples):.6f}",
                'recon': f"{total_recon / max(1, total_samples):.6f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
            })
            # Early stopping (batch-level, compare running avg total loss)
            running_avg_loss = total_loss / max(1, total_samples)
            if hasattr(self, 'min_loss') and running_avg_loss < self.min_loss:
                print(f"[EARLY STOP] Running avg loss {running_avg_loss:.4f} < min_loss ({self.min_loss}), stopping epoch early.")
                early_stop = True
                break
        epoch_avg_loss = total_loss / max(1, total_samples)
        epoch_avg_kl = total_kl / max(1, total_samples)
        epoch_avg_recon = total_recon / max(1, total_samples)
        stats = {
            'epoch': epoch + 1,
            'kl_loss': kl_loss_values.mean(),
            'recon_loss': recon_loss_values.mean(),
            'loss': loss_values.mean(),
            'epoch_avg_loss': epoch_avg_loss,
            'epoch_avg_kl': epoch_avg_kl,
            'epoch_avg_recon': epoch_avg_recon,
        }
        return stats, early_stop

    def fit(self, model, train_loader, val_loader, logger=None, epochs=10, lr=None, scheduler=None, checkpoint_dir=None, start_epoch=0, min_loss=0.1):
        import os
        import sys
        single_batch_mode = '--single_batch' in sys.argv
        self.min_loss = min_loss  # Store for use in _train_epoch
        if scheduler is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr or self.config.lr_start)
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.config.lr_factor, patience=self.config.lr_patience, min_lr=self.config.lr_end)
        else:
            optimizer = scheduler.optimizer
        # Print the first input batch and model output vectors (logits) once at the start
        first_batch = next(iter(train_loader))
        # Move all tensors in first_batch to model.device
        first_batch = [b.to(model.device) for b in first_batch]
        print(f"[DEBUG] First input batch (token tensors) on device {model.device}:")
        print(first_batch)
        with torch.no_grad():
            kl_loss, recon_loss = model(first_batch)
            # If recon_loss is a tensor, print its shape and a few values
            if hasattr(recon_loss, 'shape'):
                print(f"[DEBUG] Model output recon_loss shape: {recon_loss.shape}")
                print(f"[DEBUG] Model output recon_loss (first 5 values): {recon_loss.flatten()[:5]}")
            else:
                print(f"[DEBUG] Model output recon_loss: {recon_loss}")
        for epoch in range(start_epoch, start_epoch + epochs):
            stats, early_stop = self._train_epoch(model, epoch, train_loader, optimizer)
            if early_stop:
                print(f"[EARLY STOP] Stopping training at epoch {epoch+1} due to min_loss criterion.")
                break
            # After each epoch, evaluate on validation set to get val_loss
            model.eval()
            # Use identical structure to _train_epoch for validation
            kl_loss_values = CircularBuffer(self.config.n_last)
            recon_loss_values = CircularBuffer(self.config.n_last)
            loss_values = CircularBuffer(self.config.n_last)
            val_loss = 0.0
            val_kl = 0.0
            val_recon = 0.0
            val_samples = 0
            # Early stopping for any mode if loss < min_loss
            if stats['loss'] < min_loss:
                print(f"[EARLY STOP] Loss {stats['loss']:.4f} < min_loss ({min_loss}) at epoch {epoch+1}, stopping training.")
                break
            from tqdm import tqdm
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Val (epoch {epoch+1})")
                for batch in val_iter:
                    batch = [b.to(model.device) for b in batch]
                    batch_size = batch[0].size(0) if hasattr(batch[0], 'size') else len(batch[0])
                    kl_loss, recon_loss = model(batch)
                    loss = kl_loss + recon_loss
                    kl_loss_values.add(kl_loss.item())
                    recon_loss_values.add(recon_loss.item())
                    loss_values.add(loss.item())
                    val_loss += loss.item() * batch_size
                    val_kl += kl_loss.item() * batch_size
                    val_recon += recon_loss.item() * batch_size
                    val_samples += batch_size
                    val_iter.set_postfix({
                        'loss': f"{val_loss / max(1, val_samples):.6f}",
                        'kl': f"{val_kl / max(1, val_samples):.6f}",
                        'recon': f"{val_recon / max(1, val_samples):.6f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                    })
            mean_val_loss = val_loss / max(1, val_samples)
            mean_val_kl = val_kl / max(1, val_samples)
            mean_val_recon = val_recon / max(1, val_samples)
            scheduler.step(mean_val_loss)
            val_stats = {
                'val_loss': mean_val_loss,
                'val_kl': mean_val_kl,
                'val_recon': mean_val_recon,
                'val_loss_buffer': loss_values.mean(),
                'val_kl_buffer': kl_loss_values.mean(),
                'val_recon_buffer': recon_loss_values.mean(),
            }
            if logger is not None:
                stats.update(val_stats)
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
