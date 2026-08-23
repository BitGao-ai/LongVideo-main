"""按序列长度分桶的分布式 batch sampler（长视频训练的显存利用率优化）。

要解决的问题：collate 按**批内最大帧数**做 padding（collate.py 的 `Lmax = max(...)`）。
长视频语料里 L 的方差极大——一个 10 分钟视频（L≈600）和一个 2 小时视频（L≈4300）落进
同一批，四个样本全按 4300 分配显存，有效利用率可能只有 30%。frame_mask 保证了**计算
正确**，但显存照付。

做法（fairseq/HF 的经典 "sorted-batch shuffling"）：
    全局洗牌 → 切成大块(megabatch) → 块内按长度排序 → 切成 batch → 打乱 batch 顺序
块内排序让同批长度相近；块外仍是随机的，所以**不会退化成"按长度顺序训练"**这种会带偏
优化的采样分布。megabatch 越大分桶越紧、随机性越弱，由 bucket_multiplier 控制。

DDP 正确性（8 卡下写错会静默改变数据分布或直接挂死）：
  1) 所有 rank 用**同一个 seed+epoch** 生成**同一份** batch 列表，再按 rank 切片 →
     各 rank 数据不重叠，且每个样本每轮恰好被消费一次。
  2) batch 数被截断到 world_size 的整数倍 → **各 rank 步数严格相同**。不这样做的话，
     DDP 的 allreduce 会在快的 rank 上等一个永远不来的梯度，训练挂死。
  3) 每轮必须 set_epoch(epoch)，否则每轮洗牌相同（与 DistributedSampler 同理）。
"""
from __future__ import annotations

from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler):
    """产出 batch（索引列表）的 sampler，同批样本长度相近。

    lengths          : 每个样本的帧数（顺序与 dataset 索引一致）
    batch_size       : 单 rank 的 batch 大小
    num_replicas/rank: DDP 进程数与本进程序号；单卡传 1/0
    bucket_multiplier: megabatch = batch_size × num_replicas × 该值。越大分桶越紧、
                       随机性越弱。默认 32 在长度方差大的语料上接近最优。
    drop_last        : 丢弃最后一个不满的 batch（与 DataLoader 的 drop_last 同义）
    """

    def __init__(self, lengths: Sequence[int], batch_size: int,
                 num_replicas: int = 1, rank: int = 0, shuffle: bool = True,
                 drop_last: bool = False, seed: int = 0, bucket_multiplier: int = 32):
        if batch_size < 1:
            raise ValueError(f"batch_size 必须 ≥1，收到 {batch_size}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank={rank} 不在 [0,{num_replicas}) 内")
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
        """每轮更新洗牌种子。cycle() 会自动调用；手写训练循环必须自己调。"""
        self.epoch = int(epoch)

    def _all_batches(self) -> list[list[int]]:
        """生成**全局**（未按 rank 切分的）batch 列表。所有 rank 在同一 epoch 得到同一份。"""
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
            # 块内按长度降序：同批长度相近；降序让最长的批最先出现，OOM 会在第一步暴露，
            # 而不是训练半小时后才炸（长度升序是很常见但很坑的写法）。
            block.sort(key=lambda i: self.lengths[i], reverse=True)
            grouped.extend(block)

        batches = [grouped[s:s + self.batch_size]
                   for s in range(0, len(grouped), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if not batches:
            return []

        if self.shuffle:                     # 打乱 batch 顺序，消除"块内降序"带来的系统性排列
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 1_000_003)
            batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]

        # 截断到 world_size 整数倍：各 rank 步数必须严格相同，否则 DDP allreduce 挂死。
        # 单卡（num_replicas=1）时该截断恒为空操作，不会丢任何 batch。
        usable = (len(batches) // self.num_replicas) * self.num_replicas
        if usable == 0:                      # batch 数少于卡数：每卡各给一个（有重复，但不挂死）
            return [batches[i % len(batches)] for i in range(self.num_replicas)]
        return batches[:usable]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._all_batches()[self.rank::self.num_replicas])

    def __len__(self) -> int:
        return len(self._all_batches()) // self.num_replicas
