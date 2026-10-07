#!/usr/bin/env python3
"""Consume video features in bounded chunks, retaining only recurrent state.

Reports frontend/SSM state and emitted token counts, NOT total LLM KV memory.
Without feature files and a checkpoint this is a synthetic, untrained smoke demo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from cst_ssm.models import CSTSSMModel
from cst_ssm.utils import load_yaml, model_config_from_dict, load_checkpoint


def state_bytes(state):
    tensors = [state.seen, state.last_observed]
    for branch in state.branches:
        tensors.extend(branch.__dict__.values())
    return sum(t.numel() * t.element_size() for t in tensors)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/adaptive_cpu.yaml')
    ap.add_argument('--ckpt')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--chunk', type=int, default=16)
    ap.add_argument('--features', help='mmap .npy with shape (L,P,d)')
    ap.add_argument('--timestamps', help='mmap .npy with shape (L,) in seconds')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.chunk < 1 or args.frames < 1:
        ap.error('chunk and frames must be positive')
    if bool(args.features) != bool(args.timestamps):
        ap.error('--features and --timestamps must be supplied together')
    torch.manual_seed(args.seed)
    cfg = model_config_from_dict(load_yaml(args.config)['model'])
    if cfg.input_mode != 'feature':
        ap.error('this reader accepts feature arrays; pixel streams use model.stream_visual directly')
    model = CSTSSMModel(cfg).to(args.device).eval()
    if args.ckpt:
        load_checkpoint(model, args.ckpt, tag='stream')
    if args.features:
        if not args.features.endswith('.npy') or not args.timestamps.endswith('.npy'):
            ap.error('use separate .npy arrays for real mmap streaming, not .npz archives')
        features = np.load(args.features, mmap_mode='r')
        times = np.load(args.timestamps, mmap_mode='r')
        if features.ndim != 3 or times.shape != (features.shape[0],):
            ap.error('expected (L,P,d) features and matching (L,) timestamps')
        length = features.shape[0]
    else:
        features = times = None
        length = args.frames
        base = torch.randn(1, 1, 6, cfg.feat_dim)
    state = None
    tokens = updates = safeguard = 0
    with torch.no_grad():
        for start in range(0, length, args.chunk):
            end = min(start + args.chunk, length)
            if features is None:
                x = base + .01 * torch.randn(1, end-start, 6, cfg.feat_dim)
                ts = torch.arange(start, end, dtype=torch.float32)[None] * .5
            else:
                x = torch.from_numpy(np.array(features[start:end], copy=True))[None]
                ts = torch.from_numpy(np.array(times[start:end], copy=True))[None]
            out = model.stream_visual(dict(features=x.to(args.device), timestamps=ts.to(args.device)), state)
            state = out['state']
            tokens += int(out['visual_mask'].sum())
            updates += int(out['keep'].sum())
            safeguard += int(out['safeguard_frames'])
            del out, x, ts
    print(json.dumps(dict(mode='real_features' if features is not None else 'synthetic_demo',
                          trained=bool(args.ckpt), checkpoint=args.ckpt, config=args.config,
                          frames=length, chunk=args.chunk, seed=args.seed,
                          emitted_visual_tokens=tokens, submitted_frames=updates,
                          safeguard_only_frames=safeguard, recurrent_state_bytes=state_bytes(state),
                          includes_llm_kv=False, includes_history_index=False), indent=2))
    if not args.ckpt:
        print('Untrained model: token counts are a mechanism check, not a quality/efficiency result.')


if __name__ == '__main__':
    main()
