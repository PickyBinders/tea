from pathlib import Path
from typing import Callable

import pandas as pnd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm

class ProteinPairDataset(Dataset):
    def __init__(
        self,
        data_file,
        embeddings_prefix,
        transform: Callable = None,
    ):
        self.data = pnd.read_parquet(data_file)
        self._input_ids = torch.load(embeddings_prefix.parent / f"{embeddings_prefix.name}_tokens.pt")
        self.embeddings_prefix = embeddings_prefix
        self._embeddings = {}
        proteins = set(self.data["protein_1"]).union(self.data["protein_2"])
        for embedding_file in embeddings_prefix.parent.glob(f"{embeddings_prefix.name}_embeddings_*.pt"):
            e = torch.load(embedding_file)
            self._embeddings.update({k: v for k, v in e.items() if k in proteins})
        self._input_ids = {k: v for k, v in self._input_ids.items() if len(v) > 1 and k in self._embeddings}
        self._embeddings = {k: v for k, v in self._embeddings.items() if len(v) > 1 and k in self._input_ids}
        assert all(self._embeddings[pid].shape[0] == self._input_ids[pid].shape[0] for pid in self._embeddings)

        # Filter data to only include proteins we have tokenized sequences for
        self.data = self.data[
            self.data["protein_1"].isin(self._input_ids)
            & self.data["protein_2"].isin(self._input_ids)
        ]
        self.transform = transform
        self.proteins = set(self.data["protein_1"]).union(self.data["protein_2"])
        self.protein_pairs = list(zip(self.data["protein_1"], self.data["protein_2"]))
        self.data.set_index(["protein_1", "protein_2"], inplace=True)
        
    def __len__(self):
        return len(self.protein_pairs)

    def total_residue_triplets(self):
        return sum(len(x["index_1"]) for x in self.data["triplet_data"])

    def __getitem__(self, idx):
        protein_pair = self.protein_pairs[idx]
        residue_triplets = self.data.loc[protein_pair]["triplet_data"]
        if "index_1" not in residue_triplets:
            residue_triplets = self.data.loc[protein_pair]["triplet_data"].values[0]
        item = {
            "protein_1": protein_pair[0],
            "protein_2": protein_pair[1],
            "indices_1": residue_triplets["index_1"],
            "indices_2_positive": residue_triplets["index_2_positive"],
            "indices_2_negative": residue_triplets["index_2_negative"],
            "sequence_1": self._input_ids[protein_pair[0]],
            "sequence_2": self._input_ids[protein_pair[1]],
            "embedding_1": self._embeddings[protein_pair[0]],
            "embedding_2": self._embeddings[protein_pair[1]],
            "num_triplets": len(residue_triplets["index_1"]),
        }
        if self.transform:
            item = self.transform(item)
        return item


def collate_protein_pairs(batch):
    max_triplets = max(item["num_triplets"] for item in batch)
    batch_size = len(batch)
    indices_1 = torch.zeros((batch_size, max_triplets), dtype=torch.int64)
    indices_2_positive = torch.zeros((batch_size, max_triplets), dtype=torch.int64)
    indices_2_negative = torch.zeros((batch_size, max_triplets), dtype=torch.int64)
    proteins_1, proteins_2 = [], []
    sequences_1, sequences_2 = [], []
    embeddings_1, embeddings_2 = [], []
    num_triplets = []

    indices_1_list = [
        torch.tensor(item["indices_1"], dtype=torch.int64) for item in batch
    ]
    indices_2_positive_list = [
        torch.tensor(item["indices_2_positive"], dtype=torch.int64) for item in batch
    ]
    indices_2_negative_list = [
        torch.tensor(item["indices_2_negative"], dtype=torch.int64) for item in batch
    ]
    for i, (idx1, idx2_positive, idx2_negative) in enumerate(
        zip(indices_1_list, indices_2_positive_list, indices_2_negative_list)
    ):
        n = len(idx1)
        indices_1[i, :n] = idx1
        indices_2_positive[i, :n] = idx2_positive
        indices_2_negative[i, :n] = idx2_negative

        # Non-tensor data
        proteins_1.append(batch[i]["protein_1"])
        proteins_2.append(batch[i]["protein_2"])
        sequences_1.append(batch[i]["sequence_1"])
        sequences_2.append(batch[i]["sequence_2"])
        embeddings_1.append(batch[i]["embedding_1"])
        embeddings_2.append(batch[i]["embedding_2"])
        num_triplets.append(n)

    max_len_1 = max(seq.size(0) for seq in sequences_1)
    max_len_2 = max(seq.size(0) for seq in sequences_2)

    return {
        "protein_1": proteins_1,
        "protein_2": proteins_2,
        "sequence_1": torch.stack(
            [F.pad(seq, (0, max_len_1 - seq.size(0)), value=1) for seq in sequences_1]
        ),
        "sequence_2": torch.stack(
            [F.pad(seq, (0, max_len_2 - seq.size(0)), value=1) for seq in sequences_2]
        ),
        "embedding_1": torch.stack(
            [F.pad(emb, (0, 0, 0, max_len_1 - emb.size(0)), value=0) for emb in embeddings_1]
        ),
        "embedding_2": torch.stack(
            [F.pad(emb, (0, 0, 0, max_len_2 - emb.size(0)), value=0) for emb in embeddings_2]
        ),
        "indices_1": indices_1,
        "indices_2_positive": indices_2_positive,
        "indices_2_negative": indices_2_negative,
        "num_triplets": torch.tensor(num_triplets, dtype=torch.int64),
    }


class RandomCropProteinPair:
    def __init__(self, max_residue_triplets: int, max_seq_length: int):
        self.max_residue_triplets = max_residue_triplets
        self.max_seq_length = max_seq_length

    def get_start_end_indices(self, min_idx, max_idx, sequence_length):
        if sequence_length > self.max_seq_length:
            low = max(0, min_idx - self.max_seq_length + 1)
            high = min(sequence_length - self.max_seq_length, max_idx) + 1
            
            if low >= high:  # Invalid range
                # Fallback to a valid range
                start = max(0, sequence_length - self.max_seq_length)
            else:
                start = torch.randint(low, high, (1,)).item()
            end = start + self.max_seq_length
        else:
            start = 0
            end = sequence_length
        return start, end

    def __call__(self, item):
        min_idx1, max_idx1 = min(item["indices_1"]), max(item["indices_1"])
        min_idx2 = min(min(item["indices_2_positive"]), min(item["indices_2_negative"]))
        max_idx2 = max(max(item["indices_2_positive"]), max(item["indices_2_negative"]))

        start1, end1 = self.get_start_end_indices(
            min_idx1, max_idx1, len(item["sequence_1"])
        )
        start2, end2 = self.get_start_end_indices(
            min_idx2, max_idx2, len(item["sequence_2"])
        )
        
        # Crop sequences
        seq1_tokens = item["sequence_1"][start1:end1]
        seq2_tokens = item["sequence_2"][start2:end2]
        emb1 = item["embedding_1"][start1:end1]
        emb2 = item["embedding_2"][start2:end2]

        # Filter residue pairs that fall within the crop
        valid_triplets = sorted(
            [
                (idx1, idx2_positive, idx2_negative)
                for idx1, idx2_positive, idx2_negative in zip(
                    item["indices_1"],
                    item["indices_2_positive"],
                    item["indices_2_negative"],
                )
                if start1 <= idx1 < end1
                and start2 <= idx2_positive < end2
                and start2 <= idx2_negative < end2
            ],
            key=lambda x: (x[0], x[1]),
        )[: self.max_residue_triplets]

        # Adjust indices to be relative to the crop
        indices_1 = [idx1 - start1 for idx1, _, _ in valid_triplets]
        indices_2_positive = [
            idx2_positive - start2 for _, idx2_positive, _ in valid_triplets
        ]
        indices_2_negative = [
            idx2_negative - start2 for _, _, idx2_negative in valid_triplets
        ]

        return {
            "protein_1": item["protein_1"],
            "protein_2": item["protein_2"],
            "sequence_1": seq1_tokens,
            "sequence_2": seq2_tokens,
            "embedding_1": emb1,
            "embedding_2": emb2,
            "indices_1": indices_1,
            "indices_2_positive": indices_2_positive,
            "indices_2_negative": indices_2_negative,
            "num_triplets": len(valid_triplets),
        }


class ProteinPairDataModule(LightningDataModule):
    """
    LightningDataModule for ProteinPairDataset
    """

    def __init__(
        self,
        data_folder: Path,
        embeddings_prefix: Path,
        batch_size: int,
        num_workers: int = 0,
        max_residue_triplets: int = 100,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.data_folder = Path(data_folder)
        self.train_file = self.data_folder / "train.parquet"
        self.val_file = self.data_folder / "val.parquet"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_residue_triplets = max_residue_triplets
        self.max_seq_length = max_seq_length
        self.embeddings_prefix = embeddings_prefix
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ProteinPairDataset(
                data_file=self.train_file,
                embeddings_prefix=self.embeddings_prefix,
                transform=RandomCropProteinPair(
                    self.max_residue_triplets, self.max_seq_length
                ),
            )
            print(
               f"Train dataset has {self.train_dataset.total_residue_triplets()} triplets across {len(self.train_dataset)} protein pairs and {len(set(self.train_dataset.proteins))} unique proteins"
            )
            self.val_dataset = ProteinPairDataset(
                data_file=self.val_file,
                embeddings_prefix=self.embeddings_prefix,
                transform=RandomCropProteinPair(
                    self.max_residue_triplets, self.max_seq_length
                ),
            )
            print(
               f"Validation dataset has {self.val_dataset.total_residue_triplets()} triplets across {len(self.val_dataset)} protein pairs and {len(set(self.val_dataset.proteins))} unique proteins"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_protein_pairs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_protein_pairs,
        )