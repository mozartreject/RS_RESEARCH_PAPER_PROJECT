# Data loader utilities
"""
utils/data_loader.py — Data loading and BPR negative sampling.

Paper Section 4.1:
    - Data format: each line = "user_id item_id_1 item_id_2 ..."
    - 80/20 train/test split (already done in files)
    - Negative sampling: for each positive (u, i), sample 1 neg item j

Paper Section 4.2.3:
    - Batch size = 1024 triplets of (user, pos_item, neg_item)
"""

import os
import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset, DataLoader


class NGCFDataset:
    """
    Loads train.txt and test.txt, builds interaction matrix R for Laplacian.
    
    Attributes:
        n_users:            int, total number of users N
        n_items:            int, total number of items M
        train_data:         dict, user_id -> [item_ids]
        test_data:          dict, user_id -> [item_ids]
        interaction_matrix: scipy csr_matrix, R of shape (N, M) — Eq. 8
        train_items_set:    dict, user_id -> set(item_ids) for fast neg sampling
        n_train:            int, total training interactions = |R+|
    """

    def __init__(self, data_path, dataset_name):
        self.path = os.path.join(data_path, dataset_name)
        train_file = os.path.join(self.path, 'train.txt')
        test_file = os.path.join(self.path, 'test.txt')

        # Load both splits
        self.train_data, n_u_tr, n_i_tr = self._load_file(train_file)
        self.test_data, n_u_te, n_i_te = self._load_file(test_file)

        # Global counts (some users/items may appear only in test)
        self.n_users = max(n_u_tr, n_u_te)
        self.n_items = max(n_i_tr, n_i_te)

        # Build R (N x M) from TRAINING interactions only
        # R[u][i] = 1 if user u interacted with item i in training
        # This is used to build the Laplacian L in Equation 8
        rows, cols = [], []
        for user, items in self.train_data.items():
            for item in items:
                rows.append(user)
                cols.append(item)

        self.interaction_matrix = csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(self.n_users, self.n_items)
        )

        # Fast lookup for negative sampling
        self.train_items_set = {u: set(items) for u, items in self.train_data.items()}

        # Counts
        self.n_train = len(rows)
        self.n_test = sum(len(items) for items in self.test_data.values())

        self._print_stats()

    def _load_file(self, filepath):
        """
        Parse file where each line is: user_id item_id1 item_id2 ...
        Returns: dict, max_user_id+1, max_item_id+1
        """
        data = {}
        max_user, max_item = 0, 0

        with open(filepath, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue
                user = int(tokens[0])
                items = [int(t) for t in tokens[1:]]
                data[user] = items
                max_user = max(max_user, user)
                if items:
                    max_item = max(max_item, max(items))

        return data, max_user + 1, max_item + 1

    def _print_stats(self):
        density = self.n_train / (self.n_users * self.n_items)
        print(f"{'='*55}")
        print(f"  Dataset loaded")
        print(f"  #Users:  {self.n_users:>10,}")
        print(f"  #Items:  {self.n_items:>10,}")
        print(f"  #Train:  {self.n_train:>10,}")
        print(f"  #Test:   {self.n_test:>10,}")
        print(f"  Density: {density:>10.5f}")
        print(f"{'='*55}")


class BPRTrainDataset(Dataset):
    """
    Yields (user, pos_item, neg_item) triples for BPR loss (Eq. 11).

    Paper Section 2.4:
        O = {(u, i, j) | (u, i) in R+, (u, j) in R-}
        For each observed (u, i), sample one j that u has NOT interacted with.
    """

    def __init__(self, dataset: NGCFDataset):
        self.n_items = dataset.n_items
        self.train_items_set = dataset.train_items_set

        # Flatten training data into (user, item) pairs
        users, pos_items = [], []
        for user, items in dataset.train_data.items():
            for item in items:
                users.append(user)
                pos_items.append(item)

        self.users = np.array(users, dtype=np.int64)
        self.pos_items = np.array(pos_items, dtype=np.int64)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]

        # Rejection sampling: pick random item until we get one user hasn't seen
        neg_item = np.random.randint(0, self.n_items)
        while neg_item in self.train_items_set.get(user, set()):
            neg_item = np.random.randint(0, self.n_items)

        return user, pos_item, neg_item


def get_dataloader(dataset: NGCFDataset, batch_size: int, num_workers: int = 4):
    """Create DataLoader for BPR training. Paper: batch_size=1024."""
    return DataLoader(
        BPRTrainDataset(dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )