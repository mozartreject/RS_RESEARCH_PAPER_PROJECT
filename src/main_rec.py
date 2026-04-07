"""
main.py — Train and evaluate NGCF.
"""
import torch
torch.set_num_threads(8)
import os
import time
import random
import numpy as np
import torch.optim as optim

from config import parse_args
from models.ngcf import NGCF
from utils.data_loader import NGCFDataset, get_dataloader
from utils.adjacency import build_normalized_laplacian
from utils.metrics import evaluate_model
from utils.early_stopping import EarlyStopping


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bpr_loss(u_emb, pos_emb, neg_emb, u_ego, pos_ego, neg_ego, reg_lambda):
    pos_scores = (u_emb * pos_emb).sum(dim=1)
    neg_scores = (u_emb * neg_emb).sum(dim=1)
    bpr = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    reg = reg_lambda * (
        u_ego.norm(2).pow(2) +
        pos_ego.norm(2).pow(2) +
        neg_ego.norm(2).pow(2)
    ) / u_emb.shape[0]

    return bpr + reg, bpr.item(), reg.item()


def get_device(s):
    if s == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(s)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # NEW: Create a unique tag for this run so files don't overwrite
    run_tag = f"ngcf_{args.dataset}_L{args.n_layers}"

    print("=" * 70)
    print("  NGCF — Neural Graph Collaborative Filtering")
    print("  Faithful PyTorch Reproduction (Wang et al., SIGIR 2019)")
    print("=" * 70)
    print(f"  Dataset:          {args.dataset}")
    print(f"  Device:           {device}")
    print(f"  Embedding size:   {args.embed_size}")
    print(f"  Layers:           {args.n_layers} x {args.layer_sizes}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  L2 reg (lambda):  {args.reg}")
    print(f"  Run Tag:          {run_tag}")
    print("=" * 70)

    print("\n[Step 1/5] Loading dataset ...")
    dataset = NGCFDataset(args.data_path, args.dataset)

    print("\n[Step 2/5] Building normalized Laplacian (Eq. 8) ...")
    laplacian = build_normalized_laplacian(
        dataset.interaction_matrix, dataset.n_users, dataset.n_items
    ).to(device)

    print("\n[Step 3/5] Initializing NGCF model ...")
    pretrain_emb = None
    if args.pretrain and os.path.exists(args.pretrain):
        pretrain_emb = torch.load(args.pretrain, map_location='cpu', weights_only=True)
        print(f"  Pre-trained embeddings: {args.pretrain}")

    model = NGCF(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embed_size=args.embed_size,
        layer_sizes=args.layer_sizes,
        laplacian=laplacian,
        node_dropout=args.node_dropout,
        mess_dropout=args.mess_dropout,
        pretrain_embeddings=pretrain_emb,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n[Step 4/5] Training (max {args.epoch} epochs) ...")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Loss':>10} {'BPR':>10} {'Reg':>10} | {'Time':>6} | {'recall@20':>10} {'ndcg@20':>10}")
    print("-" * 70)

    loader = get_dataloader(dataset, args.batch_size, num_workers=0)
    
    # UPDATED: Path includes run_tag
    stopper = EarlyStopping(
        patience=args.patience,
        save_path=os.path.join(args.save_path, f'{run_tag}_best.pth'),
        verbose=False,
    )

    training_log = []

    for epoch in range(1, args.epoch + 1):
        model.train()
        t0 = time.time()
        ep_loss, ep_bpr, ep_reg, n_batch = 0., 0., 0., 0

        for batch_users, batch_pos, batch_neg in loader:
            batch_users, batch_pos, batch_neg = batch_users.to(device), batch_pos.to(device), batch_neg.to(device)

            u_emb, p_emb, n_emb, u_ego, p_ego, n_ego = model(batch_users, batch_pos, batch_neg)
            loss, bv, rv = bpr_loss(u_emb, p_emb, n_emb, u_ego, p_ego, n_ego, args.reg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_bpr += bv
            ep_reg += rv
            n_batch += 1

        ep_time = time.time() - t0
        avg_loss, avg_bpr, avg_reg = ep_loss / n_batch, ep_bpr / n_batch, ep_reg / n_batch

        if epoch % args.eval_interval == 0:
            res = evaluate_model(model, dataset, args.Ks, device)
            recall_val, ndcg_val = res[f'recall@{args.Ks[0]}'], res[f'ndcg@{args.Ks[0]}']

            print(f"{epoch:>6} | {avg_loss:>10.5f} {avg_bpr:>10.5f} {avg_reg:>10.6f} | "
                  f"{ep_time:>5.1f}s | {recall_val:>10.4f} {ndcg_val:>10.4f}")

            training_log.append({
                'epoch': epoch, 'loss': avg_loss, 'recall@20': recall_val, 'ndcg@20': ndcg_val
            })

            if stopper.step(recall_val, epoch, model):
                print(f"\n  >> Early stopping at epoch {epoch}!")
                break

    print("-" * 70)

    print("\n[Step 5/5] Final evaluation ...")
    # UPDATED: Load from run_tag path
    ckpt = os.path.join(args.save_path, f'{run_tag}_best.pth')
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    final = evaluate_model(model, dataset, args.Ks, device)

    # UPDATED: Save log with run_tag
    log_path = os.path.join(args.save_path, f'{run_tag}_log.txt')
    os.makedirs(args.save_path, exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(f"Final Results for {run_tag}:\n{final}\n")
    
    print(f"Results saved to {log_path}")

if __name__ == '__main__':
    main()