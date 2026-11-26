"""Datasets for the full-chain EBA training route."""

from collections import Counter
from functools import partial
from itertools import accumulate
from pathlib import Path

from lightning.pytorch import LightningDataModule
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tea_train.encoding import EmbeddingCache
from tea_train.io import partition_ids, read_json, sha256_file


class FullChainDataset(Dataset):
    def __init__(self, parquet, embeddings, allowed, max_triplets=2048,
                 embedding_device="cpu"):
        self.cache = EmbeddingCache(embeddings)
        self.rows = []
        self.proteins = set()
        for row in pq.read_table(parquet).to_pylist():
            left, right = row["protein_1"], row["protein_2"]
            if left not in allowed or right not in allowed:
                raise ValueError("Triplet pair is outside its training partition")
            self.proteins.update((left, right))
            length = len(row["index_1"])
            for start in range(0, length, max_triplets):
                self.rows.append({
                    "protein_1": left,
                    "protein_2": right,
                    **{
                        key: row[key][start:start + max_triplets]
                        for key in (
                            "index_1",
                            "index_2_positive",
                            "index_2_negative",
                        )
                    },
                })
        ordered = sorted(
            self.proteins,
            key=lambda key: (self.cache.manifest["chains"][key]["shard"], key),
        )
        self.embeddings = {
            key: self.cache[key].to(embedding_device) for key in ordered
        }

    def family_balanced_weights(self, mapping):
        """Return pair weights for uniform-family, then uniform-pair sampling."""
        chain_families = {
            key: set(value["families"])
            for key, value in mapping["chains"].items()
        }
        pair_families = {}
        chunks_per_pair = Counter()
        for row in self.rows:
            pair = (row["protein_1"], row["protein_2"])
            shared = chain_families[pair[0]] & chain_families[pair[1]]
            if not shared:
                raise ValueError(f"Training pair has no shared SCOP family: {pair}")
            pair_families[pair] = shared
            chunks_per_pair[pair] += 1
        family_pairs = Counter(
            family for shared in pair_families.values() for family in shared
        )
        return torch.tensor([
            sum(1 / family_pairs[family] for family in pair_families[
                (row["protein_1"], row["protein_2"])
            ]) / chunks_per_pair[(row["protein_1"], row["protein_2"])]
            for row in self.rows
        ], dtype=torch.double)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            **row,
            "embedding_1": self.embeddings[row["protein_1"]],
            "embedding_2": self.embeddings[row["protein_2"]],
        }


def collate_pairs(rows, packed=False):
    output = {
        key: [row[key] for row in rows]
        for key in ("protein_1", "protein_2")
    }
    lengths = {}
    for key in ("embedding_1", "embedding_2"):
        values = [row[key] for row in rows]
        lengths[key] = list(map(len, values))
        output[f"{key}_length"] = torch.tensor(lengths[key], dtype=torch.long)
        output[key] = torch.cat(values) if packed else pad_sequence(values, batch_first=True)
    offsets = {
        key: list(accumulate(values, initial=0))[:-1]
        for key, values in lengths.items()
    }
    for source, target, embedding in (
        ("index_1", "indices_1", "embedding_1"),
        ("index_2_positive", "indices_2_positive", "embedding_2"),
        ("index_2_negative", "indices_2_negative", "embedding_2"),
    ):
        values = [torch.tensor(row[source], dtype=torch.long) for row in rows]
        output[target] = (torch.cat([
            value + offset for value, offset in zip(values, offsets[embedding])
        ]) if packed else pad_sequence(values, batch_first=True))
    triplet_counts = [len(row["index_1"]) for row in rows]
    output["num_triplets"] = torch.tensor(triplet_counts, dtype=torch.long)
    output["batch_triplets"] = sum(triplet_counts)
    output["packed"] = packed
    return output


class FullChainDataModule(LightningDataModule):
    def __init__(
        self,
        data_folder: str,
        embeddings: str,
        splits: str,
        outer_fold: int | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        max_triplets: int = 2048,
        embedding_device: str = "cpu",
        packed_batches: bool = False,
        sampling: str = "pair",
        mapping: str | None = None,
    ):
        super().__init__()
        self.data_folder = Path(data_folder)
        self.embeddings = Path(embeddings)
        self.splits = read_json(splits)
        self.outer_fold = outer_fold
        self.full_data = self.splits.get("purpose") == "full-dataset-training"
        self.fitting_ids = partition_ids(self.splits, "fit", outer_fold)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_triplets = max_triplets
        if embedding_device not in ("cpu", "cuda"):
            raise ValueError("embedding_device must be cpu or cuda")
        if embedding_device == "cuda" and num_workers:
            raise ValueError("GPU-resident embeddings require num_workers=0")
        self.embedding_device = embedding_device
        self.packed_batches = packed_batches
        if sampling not in ("pair", "family-balanced"):
            raise ValueError("sampling must be pair or family-balanced")
        if sampling == "family-balanced" and mapping is None:
            raise ValueError("family-balanced sampling requires a SCOP mapping")
        self.sampling = sampling
        self.mapping_path = Path(mapping) if mapping is not None else None
        self.mapping = read_json(mapping) if mapping is not None else None

    def setup(self, stage=None):
        cache_path = self.embeddings / "manifest.json"
        cache = read_json(cache_path)
        cache_hash = sha256_file(cache_path)
        subsets = ("fit",) if self.full_data else ("fit", "validation")
        self.datasets = {}
        for subset in subsets:
            path = self.data_folder / f"{subset}.parquet"
            metadata = read_json(path.with_suffix(".parquet.json"))
            if metadata["teacher_id"] != cache["encoder_id"]:
                raise ValueError("EBA and head training must use the same encoder")
            if metadata.get("embedding_cache_sha256") != cache_hash:
                raise ValueError("EBA and head training must use the same embedding cache")
            allowed = set(partition_ids(self.splits, subset, self.outer_fold))
            self.datasets[subset] = FullChainDataset(
                path, self.embeddings, allowed, self.max_triplets,
                self.embedding_device,
            )

    def train_dataloader(self):
        sampler = None
        if self.sampling == "family-balanced":
            sampler = WeightedRandomSampler(
                self.datasets["fit"].family_balanced_weights(self.mapping),
                num_samples=len(self.datasets["fit"]), replacement=True,
            )
        return self._loader("fit", shuffle=sampler is None, sampler=sampler)

    def val_dataloader(self):
        if self.full_data:
            return []
        return self._loader("validation")

    def _loader(self, subset, *, shuffle=False, sampler=None):
        return DataLoader(
            self.datasets[subset],
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=partial(collate_pairs, packed=self.packed_batches),
        )
