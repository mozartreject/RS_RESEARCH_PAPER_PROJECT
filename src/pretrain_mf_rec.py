"""
pretrain_mf.py — Pre-train Matrix Factorization embeddings for NGCF.

Paper Section 4.2.3, Footnote 4:
    "We train MF from scratch, and use the trained MF embeddings to
     initialize NeuMF, GC-MC, PinSage, and NGCF to speed up and
     stabilize the training process."
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import parse_args
from utils.data_loader import NGCFDataset, BPRTrainDataset


class MF(nn.Module):
    """Basic Matrix Factorization with BPR loss."""

    def __init__(self, n_users, n_items, embed_size):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding = nn.Embedding(n_users + n_items, embed_size)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, users, pos_items, neg_items):
        u_emb = self.embedding.weight[users]
        pos_emb = self.embedding.weight[self.n_users + pos_items]
        neg_emb = self.embedding.weight[self.n_users + neg_items]
        return u_emb, pos_emb, neg_emb


def bpr_loss(u_emb, pos_emb, neg_emb, reg_lambda):
    pos_scores = (u_emb * pos_emb).sum(dim=1)
    neg_scores = (u_emb * neg_emb).sum(dim=1)
    bpr = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    reg = reg_lambda * (
        u_emb.norm(2).pow(2) +
        pos_emb.norm(2).pow(2) +
        neg_emb.norm(2).pow(2)
    ) / u_emb.shape[0]
    return bpr + reg, bpr.item(), reg.item()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("  MF Pre-training for NGCF")
    print(f"  Dataset: {args.dataset}")
    print("=" * 60)

    dataset = NGCFDataset(args.data_path, args.dataset)

    model = MF(dataset.n_users, dataset.n_items, args.embed_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loader = DataLoader(
        BPRTrainDataset(dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    mf_epochs = 200
    best_loss = float('inf')
    save_path = os.path.join(args.save_path, f'mf_{args.dataset}_pretrain.pth')
    os.makedirs(args.save_path, exist_ok=True)

    print(f"\nTraining MF for {mf_epochs} epochs...")
    print("-" * 50)

    for epoch in range(1, mf_epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss, n_batch = 0., 0

        for users, pos, neg in loader:
            u_emb, p_emb, n_emb = model(users, pos, neg)
            loss, _, _ = bpr_loss(u_emb, p_emb, n_emb, 1e-5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batch += 1

        avg_loss = ep_loss / n_batch
        ep_time = time.time() - t0

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>4d} | loss {avg_loss:.5f} | {ep_time:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.embedding.weight.data, save_path)

    print("-" * 50)
    print(f"\nPre-trained embeddings saved to: {save_path}")
    print(f"Shape: ({dataset.n_users + dataset.n_items}, {args.embed_size})")
    print(f"Best loss: {best_loss:.5f}")


if __name__ == '__main__':
    main()