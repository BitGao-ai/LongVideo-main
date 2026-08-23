#!/usr/bin/env python3
"""
benchmarks/materialize.py — 把 CSG/VFR 的 input_grid **在真实视频上重采样**成特征缓存 + 推理 manifest。

csg_build/vfr_build 只产出 `input_grid`（模型可见的帧时间戳，可非均匀/突发），并未触碰视频像素。
本脚本补上"grid → 真实视频取那几帧 → Qwen3-VL 特征 → .npz/.npy"这一步，产出可直接喂模型的推理集：

    grid(秒) --round(t·fps)--> frame_idx --decord 精确取帧--> 视觉塔 --> feats[L,P,d]
    时间戳**原样用 grid**（保住 VFR 的非均匀 Δt，这正是基准的实验变量），不重算。

关键复用：`Qwen3VLVisionFeatureExtractor.encode_video_at`（按 frame_idx 精确取帧、每帧独立时间戳）。
公平性：同一 (video, grid) 只物化一次、跨 query 共享 feature_ref——所有对照方法看**完全相同的帧**，
        差异只来自推理时"是否用真实 Δt"，与 vfr_build 的等预算/唯一网格约定一致。

用法：
    # 真实：需要 video_id→视频文件（--video-root 按 {id}{ext} 或 --video-map 显式）
    python materialize.py --manifest csg_delta2.jsonl --video-root /data/charades \
        --model Qwen/Qwen3-VL-4B-Instruct --patches 9 --out csg_feats --out-manifest csg_infer.jsonl
    # 无 GPU/decord/transformers 先跑通逻辑（合成特征，时间戳=grid）
    python materialize.py --manifest csg_demo.jsonl --dry-run --out /tmp/f --out-manifest /tmp/csg_infer.jsonl

下游：用模型对 csg_infer.jsonl 逐条推理（连续查询在帧间任意时刻求边界）→ pred.jsonl（{query_id,pred_start,pred_end}）
      → `python csg_eval.py --manifest csg_delta2.jsonl --pred pred.jsonl`。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                              # 本地 common
sys.path.insert(0, os.path.dirname(_HERE))             # 仓库根，供 cst_ssm 导入
from common import read_jsonl, write_jsonl             # noqa: E402


# ------------------------------ 纯函数（可单测，无重依赖）------------------------------
def safe_id(query_id: str) -> str:
    """query_id（含 @ . 等）→ 文件名安全串。q0001@d2.0 → q0001_d2.0。"""
    return re.sub(r"[^0-9A-Za-z._-]", "_", str(query_id))


def grid_to_frame_idx(grid, fps: float, n_frames: int | None = None) -> list:
    """grid 秒级时间戳 → 视频帧索引 round(t·fps)，clamp 到 [0, n_frames-1]。

    注：grid 比 1/fps 更密时，相邻时间戳可能映射到同一帧（物理上采样精度受 fps 限制）——
    此时特征会重复但**时间戳仍取 grid 的原值**（保 Δt 语义），下游连续查询据此在帧间求值。
    """
    idx = [int(round(float(t) * float(fps))) for t in grid]
    if n_frames is not None:
        idx = [min(max(0, i), n_frames - 1) for i in idx]
    else:
        idx = [max(0, i) for i in idx]
    return idx


def infer_row(r: dict, feature_ref: str) -> dict:
    """由 CSG/VFR 行 + 特征引用构造推理 manifest 行（兼容 VideoTemporalDataset + 携带评测字段）。"""
    row = dict(
        video_id=r["video_id"], query_id=r["query_id"], feature_ref=feature_ref,
        prompt=f'<video> {r.get("query", "")}'.strip(),
        answer=f'from {float(r["gt_start"]):.3f}s to {float(r["gt_end"]):.3f}s',
        task_type="grounding",
        gt_start=round(float(r["gt_start"]), 3), gt_end=round(float(r["gt_end"]), 3),
        duration=round(float(r["duration"]), 3), split=r.get("split", "test"),
    )
    for k in ("delta", "floor", "level", "cv"):        # 透传评测字段（不参与推理，供追溯）
        if k in r:
            row[k] = r[k]
    return row


def resolve_video_path(video_id: str, root: str, ext: str, vmap: dict | None):
    if vmap and video_id in vmap:
        return vmap[video_id]
    if root:
        return os.path.join(root, f"{video_id}{ext}")
    return None


# ------------------------------ 取帧/元数据（decord，惰性）------------------------------
def make_decord_backend():
    """返回 (reader(path,idx)->frames, meta(path)->(n_frames,fps))，共享 VideoReader 缓存。"""
    cache: dict = {}

    def _vr(path):
        vr = cache.get(path)
        if vr is None:
            import decord  # type: ignore
            decord.bridge.set_bridge("native")
            vr = cache[path] = decord.VideoReader(path)
        return vr

    def reader(path, frame_idx):
        vr = _vr(path)
        batch = vr.get_batch([int(i) for i in frame_idx]).asnumpy()
        return [batch[i] for i in range(batch.shape[0])]

    def meta(path):
        vr = _vr(path)
        return len(vr), float(vr.get_avg_fps() or 0.0)

    return reader, meta


# ------------------------------ 保存 ------------------------------
def save_features(out_dir: str, sid: str, feats: np.ndarray, ts: np.ndarray, fmt: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = np.asarray(ts, dtype=np.float32)
    if fmt == "npy":
        np.save(os.path.join(out_dir, sid + ".npy"), feats)
        np.save(os.path.join(out_dir, sid + ".ts.npy"), ts)
        return sid + ".npy"
    np.savez(os.path.join(out_dir, sid + ".npz"), features=feats, timestamps=ts)
    return sid + ".npz"


# ------------------------------ 主流程 ------------------------------
def run(args):
    rows = read_jsonl(args.manifest)
    vmap = None
    if args.video_map:
        vmap = {r["video_id"]: r["path"] for r in read_jsonl(args.video_map)}

    ex = None
    reader = meta = None
    if not args.dry_run:
        sys.path.insert(0, os.path.dirname(_HERE))
        from cst_ssm.integrations.qwen3_vl import Qwen3VLVisionFeatureExtractor
        ex = Qwen3VLVisionFeatureExtractor.from_pretrained(args.model)
        reader, meta = make_decord_backend()

    os.makedirs(os.path.dirname(args.out_manifest) or ".", exist_ok=True)
    grid_cache: dict = {}                              # (video_id, grid元组) → feature_ref，跨 query 共享
    out_rows, n_feat, n_share, n_fail = [], 0, 0, 0
    fail_log = []

    for r in rows:
        grid = [float(t) for t in r["input_grid"]]
        key = (r["video_id"], tuple(grid))
        if key in grid_cache:                          # 同 (视频,网格) 复用，保证方法间看同一批帧
            out_rows.append(infer_row(r, grid_cache[key])); n_share += 1
            continue
        sid = safe_id(r["query_id"])
        try:
            if args.dry_run:
                fps = float(r.get("fps", 30.0))
                _ = grid_to_frame_idx(grid, fps)       # 走一遍映射逻辑（验证）
                L = len(grid)
                # 种子用 sha1 而非 abs(hash(sid))：str 的内建 hash 每进程带随机盐
                # （PYTHONHASHSEED），同一 sid 两次运行拿到不同种子，dry-run 产物不可复现。
                seed = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16)
                feats = np.random.default_rng(seed).standard_normal(
                    (L, args.patches, args.out_hidden)).astype(np.float16)
                ts = np.asarray(grid, dtype=np.float32)
            else:
                path = resolve_video_path(r["video_id"], args.video_root, args.video_ext, vmap)
                if not path or not os.path.exists(path):
                    raise FileNotFoundError(f"未找到视频 {r['video_id']} → {path}")
                n_frames, real_fps = meta(path)
                fps = real_fps or float(r.get("fps", 30.0))
                frame_idx = grid_to_frame_idx(grid, fps, n_frames)
                feats, ts = ex.encode_video_at(path, frame_idx, np.asarray(grid, np.float32),
                                               target_patches=args.patches, reader=reader)
                feats = feats.astype(np.float16)
            ref_name = save_features(args.out, sid, feats, ts, args.format)
            feature_ref = os.path.join(os.path.basename(args.out), ref_name) if args.ref_prefix else ref_name
            grid_cache[key] = feature_ref
            out_rows.append(infer_row(r, feature_ref)); n_feat += 1
            if n_feat % 200 == 0:
                print(f"[materialize] 已物化 {n_feat}（最近 {sid}: L={len(ts)}）")
        except Exception as e:
            n_fail += 1
            fail_log.append({"query_id": r.get("query_id"), "error": str(e)})

    write_jsonl(args.out_manifest, out_rows)
    if fail_log:
        write_jsonl(args.out_manifest + ".failed", fail_log)
    print(f"[materialize] 物化特征 {n_feat} 条 + 共享复用 {n_share} 条 → {args.out}")
    print(f"[materialize] 推理 manifest {len(out_rows)} 行 → {args.out_manifest}"
          f"{'；失败 %d（见 .failed）' % n_fail if n_fail else ''}")
    if not args.dry_run:
        print("[materialize] 下一步：模型对推理 manifest 出 pred.jsonl → csg_eval/vfr_eval 评测")


def main():
    ap = argparse.ArgumentParser(description="CSG/VFR grid → 真实视频重采样特征 + 推理 manifest")
    ap.add_argument("--manifest", required=True, help="csg_*/vfr_*.jsonl（含 input_grid）")
    ap.add_argument("--out", default="bench_feats", help="特征输出目录")
    ap.add_argument("--out-manifest", required=True, help="推理 manifest 输出 jsonl")
    ap.add_argument("--video-root", default="", help="视频根目录（按 {video_id}{ext} 拼路径）")
    ap.add_argument("--video-ext", default=".mp4")
    ap.add_argument("--video-map", default=None, help="video_id→path 的 jsonl（覆盖 root 拼接）")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--patches", type=int, default=9)
    ap.add_argument("--out-hidden", type=int, default=3584, help="dry-run 合成维度（真实由视觉塔决定）")
    ap.add_argument("--format", default="npz", choices=["npz", "npy"], help="npy=.npy+.ts.npy(真mmap)")
    ap.add_argument("--ref-prefix", action="store_true", help="feature_ref 前加输出目录名")
    ap.add_argument("--dry-run", action="store_true", help="合成特征跑通逻辑（无需 GPU/decord/transformers）")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
