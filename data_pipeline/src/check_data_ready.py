#!/usr/bin/env python3
"""Training readiness gate: verify processed data with actionable diagnostics.

Layers: L0 manifest basics, L1 per-row deep scan, L2 config alignment,
L3 loader smoke test. Any hard error exits non-zero.

Usage:
    python -m data_pipeline.src.check_data_ready \
        --manifest data/manifests/train.jsonl --data-root data \
        --config configs/default.yaml --pipeline-config data_pipeline/configs/pipeline.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


def _load_feat_ts_closed(ref: str, root: str):
    """Load features and timestamps, closing .npz handles promptly."""
    path = os.path.join(root, ref)
    if ref.endswith(".npy"):
        feats = np.load(path, mmap_mode="r")
        ts_path = path[:-4] + ".ts.npy"
        if not os.path.exists(ts_path):
            raise FileNotFoundError(f"missing timestamp sidecar {ts_path}")
        ts = np.load(ts_path)
        return feats, np.asarray(ts, dtype=np.float32)
    with np.load(path) as z:
        if "features" not in z.files or "timestamps" not in z.files:
            raise KeyError(f"{path} lacks features/timestamps keys (has {z.files})")
        feats = np.array(z["features"])
        ts = np.asarray(z["timestamps"], dtype=np.float32)
    return feats, ts


def _count_video_tokens(prompt: str) -> int:
    return (prompt or "").count("<video>")


def check_manifest_basic(manifest: str) -> tuple[list, list]:
    errs, warns = [], []
    if not os.path.exists(manifest):
        return [f"manifest missing: {manifest}"], []
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                errs.append(f"row {i} invalid JSON: {e}")
                continue
            rows.append((i, r))
    if not rows:
        return ["manifest is empty (0 rows)"], []
    for i, r in rows:
        if not r.get("video_id"):
            errs.append(f"row {i} missing video_id")
        if not r.get("feature_ref") and not r.get("frame_dir"):
            errs.append(f"row {i} {r.get('video_id')}: no feature_ref or frame_dir")
        if "prompt" not in r:
            warns.append(f"row {i} {r.get('video_id')}: missing prompt key")
    seen: dict = {}
    for i, r in rows:
        key = (r.get("video_id"), r.get("query_id"), r.get("prompt"))
        if key in seen:
            warns.append(f"row {i} duplicates row {seen[key]} {key[:2]}")
        else:
            seen[key] = i
    return errs, warns


def check_rows_deep(manifest: str, root: str, placeholder_mode: str,
                    max_frames: int, feat_dim: int | None,
                    strict_grounding: bool, verbose: bool) -> tuple[list, list, dict]:
    try:
        from .validate_dataset import check_row
    except ImportError:
        from data_pipeline.src.validate_dataset import check_row

    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    cfg = dict(check_monotonic_ts=True, check_placeholder_align=True,
               check_dim_consistency=feat_dim is not None, warn_constant_dt=True,
               placeholder_mode=placeholder_mode, max_frames=max_frames,
               feat_dim=feat_dim, strict=True)
    if feat_dim is None:
        for r in rows:
            try:
                feats, _ = _load_feat_ts_closed(r.get("feature_ref", ""), root)
                cfg["feat_dim"] = int(feats.shape[-1])
                print(f"[gate/L1] inferred feat_dim={cfg['feat_dim']} from features")
                break
            except Exception:
                continue

    errs, warns = [], []
    dims, patches = set(), set()
    for i, r in enumerate(rows):
        ref = r.get("feature_ref")
        try:
            base_errs, d, _p = check_row(r, root, cfg)
        except Exception as e:
            errs.append(f"row {i} {r.get('video_id')}: checker error {e}")
            continue
        for e in base_errs:
            (warns if e.startswith("WARN:") else errs).append(f"row {i} {r.get('video_id')}: {e}")
        if d is not None:
            dims.add(d)
        if not ref:
            continue
        try:
            feats, ts = _load_feat_ts_closed(ref, root)
        except Exception:
            continue
        try:
            if getattr(feats, "ndim", 0) != 3:
                errs.append(f"row {i} {r.get('video_id')}: expected [L,P,d], "
                            f"got shape={tuple(feats.shape)}")
                continue
            L, P, _d = (int(x) for x in feats.shape)
            patches.add(P)
            if L == 0:
                errs.append(f"row {i} {r.get('video_id')}: empty features (L=0)")
            if not np.all(np.isfinite(ts)):
                errs.append(f"row {i} {r.get('video_id')}: timestamps contain NaN/Inf")
            elif len(ts) and float(ts[0]) < -1e-6:
                errs.append(f"row {i} {r.get('video_id')}: ts[0]={ts[0]:.3f}<0")
            nf = r.get("n_frames")
            if nf is not None and int(nf) != L:
                warns.append(f"row {i} {r.get('video_id')}: n_frames={nf} != actual L={L}")
            if "duration" in r and len(ts):
                try:
                    if abs(float(r["duration"]) - float(ts[-1])) > 1.0:
                        warns.append(f"row {i} {r.get('video_id')}: duration={r['duration']} "
                                     f"vs ts[-1]={float(ts[-1]):.2f} differs by >1s")
                except (TypeError, ValueError):
                    pass
            if os.path.isabs(ref):
                warns.append(f"row {i} {r.get('video_id')}: absolute feature_ref breaks on move")
            if ref.endswith(".npz"):
                warns.append(f"row {i} {r.get('video_id')}: still .npz; convert with npz_to_npy")
        finally:
            try:
                if hasattr(feats, "_mmap") and feats._mmap is not None:
                    feats._mmap.close()
            except Exception:
                pass

    if len(dims) > 1:
        errs.append(f"inconsistent feat_dim across rows {sorted(dims)}: mixed backbones")
    if len(patches) > 1:
        errs.append(f"inconsistent P across rows {sorted(patches)}: mixed --patches")
    if strict_grounding:
        upgrade = [w for w in warns if "grounding" in w and "outside" in w]
        for w in upgrade:
            warns.remove(w)
            errs.append(w + " (--strict-grounding)")

    info = dict(n_rows=len(rows), feat_dim=cfg.get("feat_dim"), patches=sorted(patches))
    return errs, warns, info


def check_config_align(info: dict, train_cfg_path: str | None,
                       pipeline_cfg_path: str | None,
                       cli_max_frames: int | None, cli_placeholder: str | None) -> tuple[list, list, dict]:
    errs, warns = [], []
    resolved: dict = dict(placeholder_mode=cli_placeholder or "single",
                          max_frames=cli_max_frames or 8192)
    train_data: dict = {}
    if train_cfg_path:
        try:
            from cst_ssm.utils import load_yaml
            y = load_yaml(train_cfg_path)
            train_data = y.get("data") or {}
            if train_data.get("max_frames"):
                resolved["max_frames"] = int(train_data["max_frames"])
            if cli_max_frames and int(train_data.get("max_frames", cli_max_frames)) != cli_max_frames:
                errs.append(f"CLI --max-frames={cli_max_frames} conflicts with "
                            f"{train_cfg_path} data.max_frames={train_data.get('max_frames')}")
        except Exception as e:
            errs.append(f"train config {train_cfg_path} unreadable: {e}")
    if pipeline_cfg_path and os.path.exists(pipeline_cfg_path):
        try:
            from cst_ssm.utils import load_yaml
            yp = load_yaml(pipeline_cfg_path)
            samp = (yp.get("sampler") or {})
            mani = (yp.get("manifest") or {})
            if samp.get("max_frames") and int(samp["max_frames"]) != resolved["max_frames"]:
                errs.append(f"pipeline sampler.max_frames={samp['max_frames']} != "
                            f"training max_frames={resolved['max_frames']}")
            if cli_placeholder is None and mani.get("placeholder_mode"):
                resolved["placeholder_mode"] = str(mani["placeholder_mode"])
            elif cli_placeholder and mani.get("placeholder_mode") and \
                    cli_placeholder != str(mani["placeholder_mode"]):
                warns.append(f"placeholder mode CLI={cli_placeholder} vs "
                             f"pipeline.yaml={mani['placeholder_mode']}")
            ext_p = (yp.get("extract") or {}).get("patches")
            if ext_p and info.get("patches") and len(info["patches"]) == 1 \
                    and int(ext_p) != info["patches"][0]:
                warns.append(f"pipeline extract.patches={ext_p} vs actual P={info['patches'][0]}")
        except Exception as e:
            warns.append(f"pipeline config {pipeline_cfg_path} unreadable, skipping: {e}")
    mt = train_data.get("max_text_len")
    if mt is not None and int(mt) < resolved["max_frames"] + 512:
        errs.append(f"{train_cfg_path} data.max_text_len={mt} < max_frames({resolved['max_frames']})+512; "
                    f"long samples would be silently truncated")
    return errs, warns, resolved


def check_loader_smoke(manifest: str, root: str, resolved: dict,
                       batch_size: int, batches: int, feat_dim: int | None) -> tuple[list, list]:
    errs, warns = [], []
    try:
        from cst_ssm.data import LoaderConfig, build_dataloader
    except Exception as e:
        return [f"dataloader import failed: {e}"], []
    lc = LoaderConfig(manifest=manifest, data_root=root, mode="feature",
                      max_frames=resolved["max_frames"], batch_size=batch_size,
                      num_workers=0, shuffle=False)
    if feat_dim:
        lc.feat_dim = int(feat_dim)
    try:
        loader, ds = build_dataloader(lc)
    except Exception as e:
        return [f"DataLoader construction failed: {type(e).__name__}: {e}"], []
    if len(ds) == 0:
        return ["dataset has 0 rows"], []
    if len(ds) < batch_size and lc.drop_last:
        warns.append(f"samples {len(ds)} < batch_size {batch_size} with drop_last")
    seen = 0
    total_batches = (len(ds) + batch_size - 1) // batch_size
    n_iter = max(batches, total_batches) if total_batches <= 10 else batches
    try:
        it = iter(loader)
        for _ in range(n_iter):
            try:
                b = next(it)
            except StopIteration:
                break
            seen += 1
            f = b.get("features")
            if f is not None:
                try:
                    import torch
                    with torch.no_grad():
                        if bool(torch.isnan(f).any()) or bool(torch.isinf(f).any()):
                            errs.append("first batch contains NaN/Inf")
                            break
                except ImportError:
                    pass
            ts, mask = b.get("timestamps"), b.get("frame_mask")
            if ts is not None and mask is not None:
                import torch
                for bi in range(ts.shape[0]):
                    t = ts[bi][mask[bi]].tolist()
                    if len(t) >= 2 and any(b2 < b1 for b1, b2 in zip(t, t[1:])):
                        errs.append("masked timestamps decrease inside the loader output")
                        break
            lab = b.get("labels")
            if lab is not None:
                import torch
                if bool((lab != -100).sum() == 0):
                    warns.append("batch labels are all -100 (no answers to learn)")
    except Exception as e:
        errs.append(f"loader iteration failed: {type(e).__name__}: {e}")
        return errs, warns
    if seen == 0:
        errs.append("loader yielded no batch")
        return errs, warns
    try:
        st = ds.failure_stats
        if st.get("failures", 0) or st.get("fallbacks", 0):
            errs.append(f"loader smoke saw silent replacements failures={st['failures']} "
                        f"fallbacks={st['fallbacks']}")
    except Exception:
        pass
    if not errs:
        print(f"[gate/L3] loader smoke passed: {seen} batches x {batch_size}")
    return errs, warns


def main():
    ap = argparse.ArgumentParser(description="Training readiness gate")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pipeline-config", default="data_pipeline/configs/pipeline.yaml")
    ap.add_argument("--placeholder-mode", default=None, choices=["single", "expand"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--feat-dim", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--strict-grounding", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    all_errs: list = []
    all_warns: list = []

    print("=== [gate/L0] manifest basics ===")
    e0, w0 = check_manifest_basic(a.manifest)
    all_errs += e0
    all_warns += w0
    print(f"  errors {len(e0)} warnings {len(w0)}")
    for e in e0:
        print(f"  [ERR ] {e}")
    if e0:
        print("[gate] L0 failed; fix and rerun")
        sys.exit(1)

    print("=== [gate/L2] config alignment ===")
    e2a, w2a, resolved = check_config_align(
        dict(patches=[]), a.config, a.pipeline_config, a.max_frames, a.placeholder_mode)
    print(f"  max_frames={resolved['max_frames']} placeholder={resolved['placeholder_mode']}")

    print("=== [gate/L1] per-row deep scan ===")
    e1, w1, info = check_rows_deep(a.manifest, a.data_root, resolved["placeholder_mode"],
                                   resolved["max_frames"], a.feat_dim,
                                   a.strict_grounding, a.verbose)
    all_errs += e1
    all_warns += w1
    e2b, w2b, resolved = check_config_align(
        info, a.config, a.pipeline_config, a.max_frames, a.placeholder_mode)
    for e in e2a:
        if e not in e2b:
            e2b.append(e)
    all_errs += e2b
    all_warns += w2b
    print(f"  {info['n_rows']} rows, feat_dim={info['feat_dim']} P={info['patches']}; "
          f"errors {len(e1)}, config errors {len(e2b)}")
    for e in e1 + e2b:
        print(f"  [ERR ] {e}")

    if not a.no_smoke:
        print("=== [gate/L3] loader smoke ===")
        e3, w3 = check_loader_smoke(a.manifest, a.data_root, resolved,
                                    a.batch_size, a.batches, info.get("feat_dim") or a.feat_dim)
        all_errs += e3
        all_warns += w3
        for e in e3:
            print(f"  [ERR ] {e}")
    else:
        print("=== [gate/L3] skipped (--no-smoke) ===")

    if a.verbose or all_warns:
        print(f"--- {len(all_warns)} warnings (first 10) ---")
        for w in all_warns[:10]:
            print(f"  [warn] {w}")
        if len(all_warns) > 10:
            print(f"  ... {len(all_warns) - 10} more")

    if a.report:
        os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(dict(manifest=a.manifest, data_root=a.data_root, resolved=resolved,
                           info=info, errors=all_errs, warnings=all_warns), f,
                      ensure_ascii=False, indent=2)
        print(f"[gate] report written to {a.report}")

    if all_errs:
        print(f"[gate] BLOCKED: {len(all_errs)} hard errors")
        sys.exit(1)
    print("[gate] READY for training")


if __name__ == "__main__":
    main()
