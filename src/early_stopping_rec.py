# Early stopping utilities
"""
utils/early_stopping.py

Paper Section 4.2.3:
    "premature stopping if recall@20 on the validation data
     does not increase for 50 successive epochs."
"""

import os
import torch


class EarlyStopping:
    def __init__(self, patience=50, save_path='./outputs/best.pth', verbose=True):
        self.patience = patience
        self.save_path = save_path
        self.verbose = verbose
        self.best_score = 0.0
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

    def step(self, score, epoch, model):
        """Returns True if training should stop."""
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"  * New best recall@20 = {score:.4f} at epoch {epoch}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  Early stop! Best {self.best_score:.4f} @ epoch {self.best_epoch}")
                return True
        return False