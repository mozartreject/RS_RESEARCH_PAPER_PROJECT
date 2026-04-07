# Evaluation metrics utilities
"""
utils/metrics.py — Recall@K and NDCG@K evaluation.

Paper Section 4.2.1:
    - Full ranking protocol: score ALL items, exclude training positives
    - Metrics: recall@K and ndcg@K, default K=20
    - Average over all test users

Paper footnote 3: uses corrected standard NDCG definition.
"""

import numpy as np
import torch


def recall_at_k(actual, predicted, k):
    """recall@K = |hits in top-K| / |all relevant items|"""
    if not actual:
        return 0.0
    hits = len(set(actual) & set(predicted[:k]))
    return hits / len(actual)


def ndcg_at_k(actual, predicted, k):
    """
    Standard NDCG@K with binary relevance.
        DCG@K  = sum_{i=1}^{K} rel_i / log2(i+1)
        IDCG@K = sum_{i=1}^{min(K,|rel|)} 1 / log2(i+1)
    """
    if not actual:
        return 0.0
    actual_set = set(actual)
    dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(predicted[:k]) if x in actual_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(actual_set))))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model, dataset, Ks, device, batch_size=256):
    """
    Full-ranking evaluation over all test users.

    Protocol:
        1. Compute scores for ALL items via inner product (Eq. 10)
        2. Mask training items to -inf
        3. Take top-K
        4. Compute recall@K and ndcg@K
        5. Average over all test users
    """
    model.eval()

    with torch.no_grad():
        user_emb, item_emb = model.get_all_embeddings()

    max_K = max(Ks)
    results = {f'{m}@{k}': [] for m in ('recall', 'ndcg') for k in Ks}
    test_users = list(dataset.test_data.keys())

    for start in range(0, len(test_users), batch_size):
        batch_users = test_users[start:start + batch_size]
        uids = torch.LongTensor(batch_users).to(device)

        # Inner product: y_hat(u,i) = e*_u ^T e*_i  (Eq. 10)
        scores = torch.matmul(user_emb[uids], item_emb.t())

        # Mask training positives to -inf
        for idx, u in enumerate(batch_users):
            train_items = dataset.train_items_set.get(u, set())
            if train_items:
                scores[idx, list(train_items)] = -float('inf')

        # Top-K
        _, topk_idx = torch.topk(scores, max_K, dim=1)
        topk_idx = topk_idx.cpu().numpy()

        for idx, u in enumerate(batch_users):
            actual = dataset.test_data.get(u, [])
            if not actual:
                continue
            pred = topk_idx[idx].tolist()
            for k in Ks:
                results[f'recall@{k}'].append(recall_at_k(actual, pred, k))
                results[f'ndcg@{k}'].append(ndcg_at_k(actual, pred, k))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in results.items()}