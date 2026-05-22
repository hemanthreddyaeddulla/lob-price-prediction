import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List

from src.features import OnlineFeatureEngine, N_FEATURES


class LOBDataset(Dataset):
    """One sequence per item: (features, targets, mask). I pre-compute the engineered
    features at construction time. RAM-resident but the whole dataset is small enough
    that the speedup pays for itself."""

    def __init__(
        self,
        sequences: List[dict],
        feature_engine: Optional[OnlineFeatureEngine] = None,
    ):
        self.use_features = feature_engine is not None
        self.data = []

        engine = feature_engine or OnlineFeatureEngine()

        for seq in sequences:
            states = seq["states"]
            targets = seq["targets"]
            mask = seq["mask"]

            if self.use_features:
                features = engine.process_sequence(states)
            else:
                features = states.astype(np.float32)

            self.data.append({
                "features": torch.from_numpy(features),
                "targets": torch.from_numpy(targets.astype(np.float32)),
                "mask": torch.from_numpy(mask.astype(np.float32)),
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.data[idx]
        return d["features"], d["targets"], d["mask"]


def load_sequences(parquet_path: str) -> List[dict]:
    """Group a parquet file into per-sequence dicts of raw states, targets, mask."""
    df = pd.read_parquet(parquet_path)
    feature_cols = df.columns[3:35].tolist()
    target_cols = df.columns[35:].tolist()

    sequences = []
    for seq_ix, group in df.groupby("seq_ix", sort=True):
        group = group.sort_values("step_in_seq")
        states = group[feature_cols].values
        targets = group[target_cols].values
        mask = group["need_prediction"].values
        sequences.append({
            "seq_ix": seq_ix,
            "states": states,
            "targets": targets,
            "mask": mask,
        })

    return sequences


def split_sequences_kfold(
    sequences: List[dict],
    n_folds: int = 5,
    fold: int = 0,
    seed: int = 42,
    fold_seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """K-fold split by sequence index. fold_seed must be constant across training
    seeds so the same sequences land in the same fold for every model."""
    rng = np.random.RandomState(fold_seed)
    indices = rng.permutation(len(sequences))
    fold_size = len(sequences) // n_folds

    valid_start = fold * fold_size
    valid_end = valid_start + fold_size if fold < n_folds - 1 else len(sequences)
    valid_indices = set(indices[valid_start:valid_end].tolist())

    train_seqs = [sequences[i] for i in range(len(sequences)) if i not in valid_indices]
    valid_seqs = [sequences[i] for i in range(len(sequences)) if i in valid_indices]

    return train_seqs, valid_seqs


def create_dataloaders(
    train_path: str,
    valid_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    fold: Optional[int] = None,
    n_folds: int = 5,
    seed: int = 42,
    fold_seed: int = 42,
    use_all_data: bool = False,
    use_features: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Either: (a) k-fold over the combined train+valid set, or (b) train on the
    train file and validate on the valid file. (a) is what I used for the final ensemble."""
    engine = OnlineFeatureEngine() if use_features else None

    if fold is not None or use_all_data:
        print("Loading train sequences...")
        train_seqs = load_sequences(train_path)
        print(f"  Loaded {len(train_seqs)} train sequences")
        print("Loading valid sequences...")
        valid_seqs = load_sequences(valid_path)
        print(f"  Loaded {len(valid_seqs)} valid sequences")
        all_seqs = train_seqs + valid_seqs
        print(f"  Total: {len(all_seqs)} sequences")

        if use_all_data:
            print("Using ALL data for training (no validation)")
            print("Pre-computing features...")
            train_dataset = LOBDataset(all_seqs, engine)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False,
            )
            return train_loader, None

        train_split, valid_split = split_sequences_kfold(
            all_seqs, n_folds=n_folds, fold=fold, fold_seed=fold_seed
        )
        print(f"Fold {fold}/{n_folds}: {len(train_split)} train, {len(valid_split)} valid")
    else:
        print("Loading train sequences...")
        train_split = load_sequences(train_path)
        print(f"  Loaded {len(train_split)} train sequences")
        print("Loading valid sequences...")
        valid_split = load_sequences(valid_path)
        print(f"  Loaded {len(valid_split)} valid sequences")

    print("Pre-computing features for train set...")
    train_dataset = LOBDataset(train_split, engine)
    print("Pre-computing features for valid set...")
    valid_dataset = LOBDataset(valid_split, engine)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return train_loader, valid_loader
