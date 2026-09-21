"""Length-grouped distributed batch sampler for long-video memory efficiency."""
from __future__ import annotations

from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler):
    """Batch sampler grouping similar lengths; DDP-safe sharding.

    Args:
        lengths: per-sample frame counts aligned with dataset indices.
        batch_size: per-rank batch size.
        num_replicas/rank: DDP world size and rank (1/0 for single process).
        bucket_multiplier: megabatch = batch_size x num_replicas x multiplier.
        drop_last: drop the final short batch.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int,
                 num_replicas: int = 1, rank: int = 0, shuffle: bool = True,
                 drop_last: bool = False, seed: int = 0, bucket_multiplier: int = 32):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank={rank} out of [0,{num_replicas})")
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.bucket_multiplier = max(1, int(bucket_multiplier))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _all_batches(self) -> list[list[int]]:
        """Global batch list; every rank derives the same list per epoch."""
        n = len(self.lengths)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(n, generator=g).tolist()
        else:
            order = list(range(n))

        mega = self.batch_size * self.num_replicas * self.bucket_multiplier
        grouped: list[int] = []
        for s in range(0, n, mega):
            block = order[s:s + mega]
            block.sort(key=lambda i: self.lengths[i], reverse=True)
            grouped.extend(block)

        batches = [grouped[s:s + self.batch_size]
                   for s in range(0, len(grouped), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if not batches:
            return []

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 1_000_003)
            batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]

        usable = (len(batches) // self.num_replicas) * self.num_replicas
        if usable == 0:
            return [batches[i % len(batches)] for i in range(self.num_replicas)]
        return batches[:usable]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._all_batches()[self.rank::self.num_replicas])

    def __len__(self) -> int:
        return len(self._all_batches()) // self.num_replicas
