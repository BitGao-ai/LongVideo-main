"""Decode-error 根因诊断: 不看画面、不依赖 ffprobe, 只读文件头+尾+批量规律.

用法 (拷到服务器上跑, 只读不写视频):
  python3 -m data_pipeline.src.diagnose_decode \
    --rejected data/rejected.jsonl --video-dir /workdir/data/Gwy/datasets/videos \
    --out /tmp/decode_diagnosis.jsonl
  # 只查前20个: 加 --limit 20
  # rejected.jsonl 不存在时: 直接全量扫描 --video-dir (加 --scan-all)

 verdict 含义:
  path_unknown_no_videodir               -> 没给 --video-dir, 根本没找文件, 结论不可用, 补上重跑
  path_not_found_in_dir                  -> 给了video-dir仍找不到该vid, 先核对目录是否一致
  zero_byte                              -> 下载/拷贝失败, 重拉
  html_or_json_error_page                -> 下载到的是报错页面, 重拉
  no_ftyp_no_moov / no_moov_truncated    -> 截断/未传完, 重拉该批次
  no_video_track_audio_only              -> 音频-only或无视频流, decord必挂, 属文件问题
  DECORD-ONLY-FAIL (cv2/ffprobe能解)     -> 文件大概率OK, 是decord/ffmpeg版本坑, 别重拉
  LIKELY-FILE-ISSUE                      -> 头尾有moov但decord解不出且无第二解码器佐证, 按文件坏处理
  file_ok_header + DECODABLE-NOW         -> 文件现在能解, 当初是抖动, 重跑quality_filter即可
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess

from .dist_utils import _VIDEO_EXTS, feature_stem

HEAD_N = 2 * 1024 * 1024   # 头2MB: 找 ftyp/moov/trak
TAIL_N = 4 * 1024 * 1024   # 尾4MB: moov 经常在尾部, 截断下载尾部缺失


def read_head_tail(path: str):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(HEAD_N)
        tail = b""
        if size > HEAD_N:
            f.seek(max(0, size - TAIL_N))
            tail = f.read()
        else:
            tail = head
    return size, head, tail


def classify_file(path: str) -> dict:
    """纯标准库分类, 不解码."""
    if not os.path.exists(path):
        return {"verdict": "file_missing", "action": "re-pull"}
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return {"verdict": "stat_fail", "detail": str(e), "action": "check-disk"}
    if size == 0:
        return {"verdict": "zero_byte", "size": 0, "action": "re-pull"}
    if size < 32 * 1024:
        # 太小基本不可能是正常长视频, 但仍看内容
        pass
    try:
        size2, head, tail = read_head_tail(path)
        size = size2
    except OSError as e:
        return {"verdict": "unreadable", "size": size, "detail": str(e), "action": "check-disk"}

    magic = head[:32]
    # 1) 下载到报错页?
    stripped = head.lstrip()[:16].lower()
    if stripped.startswith((b"<html", b"<!doctype", b"{\"error", b"{\"code", b"not found", b"<?xml")):
        return {"verdict": "html_or_json_error_page", "size": size,
                "head_hex": magic[:16].hex(), "action": "re-pull-batch"}
    # 2) EBML (mkv/webm) 却套 .mp4 后缀? decord 老版本可能挂
    if head[:4] == b"\x1aE\xdf\xa3":
        return {"verdict": "ebml_container", "size": size, "action": "cross-check-decoder"}
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return {"verdict": "avi_container", "size": size, "action": "cross-check-decoder"}
    if not path.lower().endswith((".mp4", ".mov", ".m4v")):
        return {"verdict": "non_mp4_container", "size": size, "action": "cross-check-decoder"}

    has_ftyp = b"ftyp" in head[:16 * 1024]
    has_moov_head = b"moov" in head
    has_moov_tail = b"moov" in tail
    has_trak = b"trak" in head or b"trak" in tail
    # 视频轨道线索: vmhd=video media header, avc1/hvc1/mp4v/av01=常见视频编码
    has_video_tag = any(t in head or t in tail for t in (b"vmhd", b"avc1", b"hvc1", b"hev1", b"mp4v", b"av01"))
    has_audio_tag = b"mp4a" in head or b"mp4a" in tail
    n_trak = head.count(b"trak") + (tail.count(b"trak") if len(tail) != len(head) else 0)

    info = {"size": size, "has_ftyp": has_ftyp,
            "has_moov_head": has_moov_head, "has_moov_tail": has_moov_tail,
            "has_trak": has_trak, "has_video_tag": has_video_tag,
            "has_audio_tag": has_audio_tag, "n_trak_hint": n_trak}

    if not has_ftyp and not has_moov_head and not has_moov_tail:
        # 连 ftyp/moov 都没有: 要么截断成纯 mdat, 要么根本不是 mp4
        return {**info, "verdict": "no_ftyp_no_moov", "action": "re-pull-batch"}
    if not has_moov_head and not has_moov_tail:
        return {**info, "verdict": "no_moov_truncated", "action": "re-pull-batch"}
    if has_moov_head or has_moov_tail:
        if not has_trak and not has_video_tag and not has_audio_tag:
            return {**info, "verdict": "moov_without_trak_suspicious", "action": "ffprobe-confirm"}
        if not has_video_tag and has_audio_tag:
            return {**info, "verdict": "no_video_track_audio_only",
                    "action": "file-issue-not-decord"}
        if not has_video_tag:
            return {**info, "verdict": "no_video_tag_found",
                    "action": "ffprobe-confirm"}
    return {**info, "verdict": "file_ok_header", "action": "cross-check-decoder"}


def try_decord(path: str) -> str:
    try:
        import decord  # type: ignore
    except Exception as e:
        return f"decord-missing: {e}"
    try:
        vr = decord.VideoReader(path)
        _ = len(vr)
        _ = vr[0].asnumpy()
        return "decord-ok"
    except Exception as e:
        return f"decord-fail: {str(e)[:200]}"


def try_cv2(path: str) -> str:
    try:
        import cv2  # type: ignore
    except Exception as e:
        return f"cv2-missing: {e}"
    try:
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened()
        if not ok:
            return "cv2-fail: not-opened"
        r, _ = cap.read()
        cap.release()
        return "cv2-ok" if r else "cv2-fail: read-false"
    except Exception as e:
        return f"cv2-fail: {str(e)[:200]}"


def try_ffprobe(path: str) -> str:
    for bin_name in ("ffprobe",):
        try:
            p = subprocess.run([bin_name, "-v", "error", "-show_entries",
                                "stream=codec_type,codec_name",
                                "-of", "default=noprint_wrappers=1", path],
                               capture_output=True, text=True, timeout=20)
            out = (p.stdout + p.stderr).strip()[:300]
            if "video" in out.lower():
                return f"ffprobe-has-video: {out[:150]}"
            return f"ffprobe-no-video: {out[:150] or 'empty'}"
        except FileNotFoundError:
            continue
        except Exception as e:
            return f"ffprobe-error: {e}"
    return "ffprobe-missing"


def diagnose_one(video_id: str, path: str, reason: str, detail: str) -> dict:
    # 没给 --video-dir 时 path 是占位符: 不能下任何文件结论
    if path.startswith("<path-unknown:"):
        return {"video_id": video_id, "path": path,
                "orig_reason": reason, "orig_detail": (detail or "")[:200],
                "verdict": "path_unknown_no_videodir", "action": "rerun-with-video-dir",
                "final": "UNKNOWN",
                "suggest": "没给--video-dir, 没查文件, 补上--video-dir重跑"}
    file_info = classify_file(path)
    v = file_info["verdict"]
    # 在给定目录下找不到文件: 先怀疑目录给错, 别直接判重拉
    if path.startswith("<not-found:") and v == "file_missing":
        file_info.update({"video_id": video_id, "path": path,
                          "orig_reason": reason, "orig_detail": (detail or "")[:200],
                          "verdict": "path_not_found_in_dir",
                          "final": "PATH-NOT-FOUND",
                          "suggest": "video-dir下找不到该vid, 先核对--video-dir是否和quality_filter跑的是同一目录/后缀, 确认文件真缺失再重拉"})
        return file_info
    # 文件层已定案的, 不再调解码器浪费时间
    if v in ("file_missing", "zero_byte", "html_or_json_error_page",
             "no_ftyp_no_moov", "no_moov_truncated", "no_video_track_audio_only"):
        file_info.update({"video_id": video_id, "path": path,
                          "orig_reason": reason, "orig_detail": (detail or "")[:200],
                          "final": "FILE-ISSUE",
                          "suggest": "重拉该批次, quality_filter判对了"})
        return file_info
    # 头部看着OK: 交叉验证是不是 decord 专属问题
    d = try_decord(path)
    c = try_cv2(path)
    f = try_ffprobe(path)
    if d == "decord-ok":
        final, suggest = "DECODABLE-NOW", "文件现在能解, 当初可能是并发/网络盘抖动, 重跑quality_filter即可"
    elif ("cv2-ok" in c) or ("ffprobe-has-video" in f):
        final, suggest = "DECORD-ONLY-FAIL", "cv2/ffprobe能解只有decord挂: decord或ffmpeg版本坑, 别重拉, 升级decord或换解码器"
    elif ("missing" in c) and ("missing" in f):
        final, suggest = "LIKELY-FILE-ISSUE", "头尾虽有moov但decord解不出且无第二解码器佐证, 按文件坏处理, 抽1个补ffprobe实锤"
    else:
        final, suggest = "FILE-ISSUE", "双解码器都解不出, 文件坏, 重拉"
    file_info.update({"video_id": video_id, "path": path,
                      "orig_reason": reason, "orig_detail": (detail or "")[:200],
                      "decord": d, "cv2": c, "ffprobe": f,
                      "final": final, "suggest": suggest})
    return file_info


def build_stem_index(video_dir: str) -> dict[str, str]:
    """video_dir 下所有视频文件的 stem -> path, 建一次复用 (避免每条 ghost 全量 walk)."""
    idx: dict[str, str] = {}
    for root, _, files in os.walk(video_dir):
        for fn in files:
            if fn.lower().endswith(_VIDEO_EXTS):
                p = os.path.join(root, fn)
                stem = feature_stem(fn)
                idx.setdefault(stem, p)
                rel = feature_stem(os.path.relpath(p, video_dir))
                idx.setdefault(rel, p)
    return idx


def resolve_path(vid: str, video_dir: str, index: dict[str, str] | None) -> str | None:
    # 快路径: 平铺或子目录还原, 只做两次 stat
    for ext in _VIDEO_EXTS:
        for name in (vid + ext, vid.replace("_", "/") + ext):
            p = os.path.join(video_dir, name)
            if os.path.exists(p):
                return p
    # 慢路径: 查预建索引 (子目录深层/后缀大小写等), 不再 walk
    if index is not None and vid in index:
        return index[vid]
    return None


def main():
    ap = argparse.ArgumentParser(description="decode_error 根因诊断 (只读)")
    ap.add_argument("--rejected", default="data/rejected.jsonl")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--out", default="/tmp/decode_diagnosis.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scan-all", action="store_true",
                    help="不读rejected, 全量扫描video-dir (慢, 抽查用limit)")
    a = ap.parse_args()

    items: list[tuple[str, str, str]] = []  # (vid, path, detail)
    stem_index: dict[str, str] | None = None
    if a.video_dir and not a.scan_all:
        # 预建一次索引供慢路径用; 目录巨大时这里是一次性 walk
        stem_index = build_stem_index(a.video_dir)
    if a.scan_all and a.video_dir:
        for root, _, files in os.walk(a.video_dir):
            for fn in sorted(files):
                if fn.lower().endswith(_VIDEO_EXTS):
                    p = os.path.join(root, fn)
                    vid = feature_stem(os.path.relpath(p, a.video_dir))
                    items.append((vid, p, ""))
    else:
        with open(a.rejected) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("reason") != "decode_error":
                    continue
                vid = r["video_id"]
                if not a.video_dir:
                    items.append((vid, f"<path-unknown:{vid}>", r.get("detail", "")))
                    continue
                cand = resolve_path(vid, a.video_dir, stem_index)
                items.append((vid, cand or f"<not-found:{vid}>", r.get("detail", "")))
    if a.limit:
        items = items[:a.limit]

    rows = [diagnose_one(vid, p, "decode_error", d) for vid, p, d in items]
    if a.out:
        d = os.path.dirname(a.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter((r["verdict"], r["final"] if "final" in r else "") for r in rows)
    print(f"[diagnose] checked {len(rows)} files -> {a.out}")
    for (v, fin), n in c.most_common():
        print(f"  {v} / {fin}: {n}")
    # 批量规律: 同 verdict 是否连续ID成片?
    vids = sorted(r["video_id"] for r in rows if r.get("final") == "FILE-ISSUE")
    print(f"  FILE-ISSUE e.g.: {vids[:8]}" + (" ..." if len(vids) > 8 else ""))


if __name__ == "__main__":
    main()
