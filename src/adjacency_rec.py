# Adjacency matrix construction utilities
"""
utils/adjacency.py — Normalized Laplacian for NGCF.

Paper Section 2.2.2, Equation 8:

    L = D^{-1/2} * A * D^{-1/2}

    A = | 0    R  |     adjacency of bipartite graph, shape (N+M, N+M)
        | R^T  0  |

    R = user-item interaction matrix, shape (N, M)
    D = diagonal degree matrix, D_tt = number of neighbors of node t

    Nonzero entry: L_{u,i} = 1 / sqrt(|N_u| * |N_i|) = p_ui from Eq. 3

    Self-loops (identity I) are NOT included in L.
    They are handled separately in the forward pass via Eq. 7: (L + I).
"""

import numpy as np
import scipy.sparse as sp
import torch


def build_normalized_laplacian(interaction_matrix, n_users, n_items):
    """
    Build L = D^{-1/2} A D^{-1/2} as a PyTorch sparse tensor.

    Args:
        interaction_matrix: scipy csr_matrix of shape (N, M)
        n_users: int, N
        n_items: int, M

    Returns:
        torch.sparse.FloatTensor of shape (N+M, N+M)
    """
    # ── Step 1: Build adjacency A ──
    # A = | 0   R  |   shape (N+M) x (N+M)
    #     | R^T 0  |
    R = interaction_matrix.tocoo()
    n_nodes = n_users + n_items

    # User-to-item edges: row=user, col=n_users+item (offset items)
    # Item-to-user edges: row=n_users+item, col=user (symmetric)
    rows = np.concatenate([R.row, R.col + n_users])
    cols = np.concatenate([R.col + n_users, R.row])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # ── Step 2: Degree vector ──
    # D_tt = number of neighbors of node t
    degrees = np.array(A.sum(axis=1)).flatten()

    # ── Step 3: D^{-1/2} ──
    # Avoid division by zero for isolated nodes
    d_inv_sqrt = np.where(degrees > 0, np.power(degrees, -0.5), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    # ── Step 4: L = D^{-1/2} A D^{-1/2} ──
    L = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()

    print(f"  Laplacian: shape={L.shape}, nnz={L.nnz:,}")

    return _to_torch_sparse(L)


def _to_torch_sparse(sp_coo):
    """Convert scipy COO matrix to PyTorch sparse tensor."""
    sp_coo = sp_coo.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.vstack([sp_coo.row, sp_coo.col]))
    values = torch.FloatTensor(sp_coo.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(sp_coo.shape)).coalesce()


def apply_node_dropout(laplacian, dropout_rate):
    """
    Node dropout (Section 2.4.2):
        Randomly drop (N+M)*p2 nodes and ALL their edges.
        Rescale surviving edges by 1/(1-p2) to maintain expectation.

    Args:
        laplacian: torch sparse tensor, L
        dropout_rate: float, p2

    Returns:
        Dropped and rescaled sparse tensor.
    """
    if dropout_rate <= 0.0:
        return laplacian

    n_nodes = laplacian.shape[0]
    indices = laplacian.indices()
    values = laplacian.values()

    # Each node survives with probability (1 - dropout_rate)
    keep_mask = torch.rand(n_nodes, device=laplacian.device) > dropout_rate

    # Edge survives only if BOTH endpoints survive
    edge_keep = keep_mask[indices[0]] & keep_mask[indices[1]]

    new_values = values[edge_keep] / (1.0 - dropout_rate)
    new_indices = indices[:, edge_keep]

    return torch.sparse_coo_tensor(
        new_indices, new_values, laplacian.shape
    ).coalesce()