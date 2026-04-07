"""
config.py — All hyperparameters from NGCF paper Section 4.2.3.
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="NGCF Reproduction (PyTorch)")
    # ── Dataset ──
    parser.add_argument('--dataset', type=str, default='gowalla')
    parser.add_argument('--data_path', type=str, default='./data/')
    # ── Model Architecture (Section 2) ──
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--layer_sizes', type=str, default='64,64,64')
    # ── Training (Section 4.2.3) ──
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--reg', type=float, default=1e-5)
    # ── Dropout (Section 2.4.2) ──
    parser.add_argument('--mess_dropout', type=float, default=0.1)
    parser.add_argument('--node_dropout', type=float, default=0.0)
    # ── Evaluation (Section 4.2.1) ──
    parser.add_argument('--Ks', type=str, default='20')
    # ── Early Stopping ──
    parser.add_argument('--patience', type=int, default=50)
    # ── Misc ──
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='./outputs/')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--pretrain', type=str, default='',
                        help='Path to pre-trained MF embeddings (.pth file)')
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()
    args.Ks = [int(k) for k in args.Ks.split(',')]
    args.layer_sizes = [int(s) for s in args.layer_sizes.split(',')]
    assert len(args.layer_sizes) == args.n_layers
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)