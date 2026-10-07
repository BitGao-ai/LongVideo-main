"""Continuous-boundary baselines using only discrete frame readouts.

These deliberately demonstrate that continuous outputs do not require continuous
SSM states. Both modes share query conditioning and input/supervision budgets.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class BoundaryRegressionHead(nn.Module):
    def __init__(self, dim: int, mode: str = 'offset'):
        super().__init__()
        if mode not in ('offset', 'direct'):
            raise ValueError('boundary regression mode must be offset or direct')
        self.mode = mode
        self.local = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 4))
        self.global_head = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def forward(self, readouts, query, timestamps, mask=None):
        valid = torch.ones_like(timestamps, dtype=torch.bool) if mask is None else mask.bool()
        if not bool(valid.any(-1).all()):
            raise ValueError('boundary regression requires a valid observed frame in every row')
        if not torch.isfinite(timestamps[valid]).all():
            raise ValueError('timestamps must be finite')
        q = query[:, None].expand_as(readouts)
        features = torch.cat((readouts, q, readouts * q), -1)
        rows = []
        for b in range(readouts.shape[0]):
            x, t = features[b, valid[b]], timestamps[b, valid[b]]
            if bool((t[1:] < t[:-1]).any()):
                raise ValueError('valid timestamps must be ordered')
            if self.mode == 'direct':
                # Continuous regression relative to the observed time span.
                relative = self.global_head(x.mean(0)).sigmoid()
                boundary = t[0] + relative * (t[-1] - t[0])
            else:
                local = self.local(x)
                weights = local[:, :2].softmax(0)
                # A frame score plus an intra-cell offset; no extra observations.
                middle = (t[:-1] + t[1:]) * .5
                left = torch.cat((t[:1], middle))
                right = torch.cat((middle, t[-1:]))
                candidates = left[:, None] + local[:, 2:].sigmoid() * (right-left)[:, None]
                boundary = (weights * candidates).sum(0)
            rows.append(boundary.sort().values)
        return torch.stack(rows)


def boundary_regression_loss(prediction, target, timestamps, mask=None):
    valid = torch.ones_like(timestamps, dtype=torch.bool) if mask is None else mask.bool()
    first = timestamps.masked_fill(~valid, torch.inf).min(-1).values
    last = timestamps.masked_fill(~valid, -torch.inf).max(-1).values
    span = (last-first).clamp_min(1e-3)
    return F.smooth_l1_loss(prediction/span[:, None], target/span[:, None])
