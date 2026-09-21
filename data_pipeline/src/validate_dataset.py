"""Pre-training gate: hard checks on timestamps, placeholders, dims, and file access."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _load_feat_ts(ref: str, root: str):
    path = os.path.join(root, ref)
    if ref.endswith(".npy"):
        feats = np.load(path, mmap_mode="r")
        ts_path = path[:-4] + ".ts.npy"
        if not os.path.exists(ts_path):
            raise FileNotFoundError(f"missing timestamp sidecar {ts_path}")
        ts = np.load(ts_path)
        return feats, np.asarray(ts, np.float32)
    z = np.load(path)
    return z["features"], np.asarray(z["timestamps"], np.float32)


def count_video_tokens(prompt: str) -> int:
    return prompt.count("<video>")


def _infer_feat_dim(rows: list, root: str):
    """Infer feature dim from the first readable entry; None when none is readable."""
    for r in rows:
        ref = r.get("feature_ref")
        if not ref:
            continue
        try:
            feats, _ = _load_feat_ts(ref, root)
            return int(feats.shape[-1])
        except Exception:
            continue
    return None


def check_row(r: dict, root: str, cfg: dict) -> tuple[list, int | None]:
    """Return (errors, feature dim); soft warnings carry a 'WARN:' prefix."""
    errs = []
    ref = r.get("feature_ref")
    if not ref:
        return ["missing feature_ref"], None
    try:
        feats, ts = _load_feat_ts(ref, root)
    except Exception as e:
        return [f"unreachable feature_ref: {e}"], None

    L = int(feats.shape[0]); d = int(feats.shape[-1])

    if cfg["check_monotonic_ts"]:
        if len(ts) != L:
            errs.append(f"ts len {len(ts)} != frames {L}")
        elif len(ts) >= 2 and not np.all(np.diff(ts) > 0):
            bad = int((np.diff(ts) <= 0).sum())
            errs.append(f"timestamps not strictly increasing ({bad} non-positive dt)")

    if cfg["check_placeholder_align"]:
        n_tok = count_video_tokens(r.get("prompt", ""))
        if cfg["placeholder_mode"] == "single":
            if n_tok != 1:
                errs.append(f"<video> count {n_tok} != 1 (single mode)")
        else:
            expect = min(L, cfg["max_frames"])
            if n_tok != expect:
                errs.append(f"<video> count {n_tok} != min(L,max_frames)={expect}")

    if cfg["check_dim_consistency"] and d != cfg["feat_dim"]:
        errs.append(f"feat_dim {d} != expected {cfg['feat_dim']}")

    if cfg["warn_constant_dt"] and len(ts) >= 3:
        dt = np.diff(ts)
        if dt.std() < 1e-4 * (dt.mean() + 1e-9):
            errs.append("WARN: near-constant dt (sampler may have degraded to uniform)")

    if r.get("task_type") == "grounding" and "gt_end" in r:
        dur = float(r.get("duration", ts[-1] if len(ts) else 0))
        if float(r["gt_end"]) > dur + 1e-3 or float(r.get("gt_start", 0)) < -1e-3:
            errs.append(f"WARN: grounding span [{r.get('gt_start')},{r['gt_end']}] outside duration {dur}")
    return errs, d


def run(args, cfg: dict):
    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if cfg["check_dim_consistency"] and cfg["feat_dim"] is None:
        d = _infer_feat_dim(rows, args.data_root)
        if d is None:
            print("[validate] --feat-dim missing and no readable feature; skipping dim check")
            cfg["check_dim_consistency"] = False
        else:
            cfg["feat_dim"] = d
            print(f"[validate] inferred expected dim={d} from features")
    n_err = n_warn = 0
    dims = set()
    for i, r in enumerate(rows):
        errs, dim = check_row(r, args.data_root, cfg)
        for e in errs:
            if e.startswith("WARN:"):
                n_warn += 1
                if args.verbose:
                    print(f"  [warn] row {i} {r.get('video_id')}: {e}")
            else:
                n_err += 1
                print(f"  [ERR ] row {i} {r.get('video_id')}: {e}")
        if dim is not None:
            dims.add(dim)

    print(f"[validate] {len(rows)} rows: {n_err} errors, {n_warn} warnings")
    if len(dims) > 1:
        print(f"  [ERR ] inconsistent feat_dim across rows: {sorted(dims)}"); n_err += 1
    if n_err == 0:
        print("[validate] all hard checks passed")
    else:
        print("[validate] hard errors found; training blocked")
        if cfg["strict"]:
            sys.exit(1)


DEFAULTS = dict(check_monotonic_ts=True, check_placeholder_align=True,
                check_dim_consistency=True, warn_constant_dt=True,
                placeholder_mode="single", max_frames=8192, feat_dim=None, strict=True)


def main():
    ap = argparse.ArgumentParser(description="Pre-training dataset gate")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--placeholder-mode", default="single", choices=["single", "expand"])
    ap.add_argument("--max-frames", type=int, default=8192)
    ap.add_argument("--feat-dim", type=int, default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    cfg = dict(DEFAULTS, placeholder_mode=a.placeholder_mode, max_frames=a.max_frames,
               feat_dim=a.feat_dim, strict=a.strict)
    run(a, cfg)


if __name__ == "__main__":
    main()
