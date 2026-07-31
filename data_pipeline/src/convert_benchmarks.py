"""四基准 → 统一 QA manifest（对齐 VideoSample + 评测所需字段）。评测集，独立于训练。

真实 per-benchmark 适配器（字段名/答案编码各不同，按官方标注归一）：
  - VideoMME  : 每问一行，`duration` 是**档位字符串**(short/medium/long) 而非秒 → 存 duration_bucket
                （GATE-4 需按 long 子集报告）；答案为字母 A–D；options 形如 "A. xxx"。
  - MLVU      : `video` 文件名；`candidates` 选项；答案字母或选项原文；`question_type` 任务类。
  - LVBench   : 超长视频；`question`(可能含选项)/`options`/`answer`/`question_type`。
  - EgoSchema : `q_uid` 既是问题也是视频键；5 选 1；答案为**0基索引**(fullset 可能无答案→-1)。

归一 schema（每行）：
  {video_id, query_id, feature_ref, prompt:"<video> Q\nA. ..\nB. ..",
   answer:"B"(字母), answer_index:1(0基), options:[原文...], task_type, [duration_bucket], bench, split}

注：本脚本只产文本侧 manifest；评测视频同样需先经 extract_features 抽特征（feature_ref 指向）。
    非 json/jsonl（如 parquet）请先转成 jsonl。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string

LETTERS = string.ascii_uppercase
_OPT_PREFIX = re.compile(r"^\(?([A-Z])[.)]\s+", re.I)      # "A. " / "A) " / "(A) "


# 每基准：字段候选（按序取首个命中）+ 整数答案基（0/1）
ADAPTERS = {
    "videomme": dict(vid=["videoID", "video_id", "video"], q=["question"],
                     opts=["options", "candidates"], ans=["answer"],
                     task=["task_type", "sub_category", "domain"],
                     bucket=["duration", "duration_category"], base=1,
                     qid=["question_id", "qid"]),
    "mlvu":     dict(vid=["video", "video_id"], q=["question"],
                     opts=["candidates", "options"], ans=["answer"],
                     task=["question_type", "task_type", "task"], bucket=[], base=1,
                     qid=["question_id", "qid"]),
    "lvbench":  dict(vid=["video_id", "video", "key"], q=["question"],
                     opts=["options", "candidates"], ans=["answer"],
                     task=["question_type", "type"], bucket=[], base=1,
                     qid=["question_id", "uid", "qid"]),
    "egoschema": dict(vid=["q_uid", "video_id", "video"], q=["question"],
                      opts=["option", "options"], ans=["answer", "correct_answer"],
                      task=["task_type"], bucket=[], base=0, qid=["q_uid"]),
    # LongVideoBench：correct_choice 是 0 基索引；duration_group∈{15,60,600,3600}(秒档)；id 为问题唯一键
    "longvideobench": dict(vid=["video_id", "video_path", "video"],
                           q=["question", "question_wo_referring_query"],
                           opts=["candidates", "options"], ans=["correct_choice", "answer"],
                           task=["question_category", "level", "topic_category"],
                           bucket=["duration_group"], base=0,
                           qid=["id", "question_id"]),
}


def _first(rec: dict, keys, default=None):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return default


def _iter_records(src: str):
    if src.endswith(".jsonl"):
        for l in open(src, encoding="utf-8"):
            if l.strip():
                yield json.loads(l)
    else:
        data = json.load(open(src, encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", data.get("questions", list(data.values())))
        yield from data


def extract_options(rec: dict, opt_keys) -> list:
    """抽取选项原文列表（去掉已有的 'A. ' 前缀）。兼容 list/dict/option_0..N 分散字段。"""
    raw = _first(rec, opt_keys)
    if raw is None:                                    # 尝试分散字段 option_0.. / a,b,c,d
        seq = []
        for i in range(8):
            v = rec.get(f"option_{i}", rec.get(f"option{i}"))
            if v is None:
                break
            seq.append(v)
        if not seq:
            for L in "abcde":
                if L in rec:
                    seq.append(rec[L])
        raw = seq or None
    if raw is None:
        return []
    if isinstance(raw, dict):                          # {"A":..,"B":..}
        raw = [raw[k] for k in sorted(raw)]
    if not isinstance(raw, list):
        raw = [raw]
    out = []
    for o in raw:
        s = str(o).strip()
        out.append(_OPT_PREFIX.sub("", s).strip())     # 去 "A. "/"A) "/"(A) " 前缀，保留原文
    return out


def normalize_answer(raw, options: list, base: int = 1):
    """把答案归一为 (letter, index0)。支持：字母 / 整数索引(base 0或1) / 选项原文匹配。

    返回 ("", -1) 表示未知（如 EgoSchema fullset 无答案）。
    """
    n = len(options)
    if raw is None or raw == "":
        return "", -1
    s = str(raw).strip()
    # 1) 单字母 —— 仅当落在选项范围内才当字母（否则可能是单字符选项原文，如 MLVU 答案"q"）
    if len(s) == 1 and s.upper() in LETTERS:
        idx = LETTERS.index(s.upper())
        if n == 0 or idx < n:
            return LETTERS[idx], idx
        # 超出选项范围 → 落到下方原文匹配
    # 2) 纯整数索引
    if s.lstrip("-").isdigit():
        v = int(s)
        idx = v - base if base else v
        if 0 <= idx < max(n, 1):
            return LETTERS[idx], idx
        # base 猜错兜底
        alt = v - (1 - base)
        if 0 <= alt < max(n, 1):
            return LETTERS[alt], alt
        return "", -1
    # 3) 选项原文匹配
    for i, o in enumerate(options):
        if s.lower() == str(o).strip().lower():
            return LETTERS[i], i
    return "", -1


def build_prompt(question: str, options: list) -> str:
    q = str(question).replace("<video>", "").strip()
    if not options:
        return f"<video> {q}"
    lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    return f"<video> {q}\n" + "\n".join(lines)


def convert_record(rec: dict, bench: str, idx: int, feature_subdir: str) -> dict | None:
    ad = ADAPTERS[bench]
    vid = _first(rec, ad["vid"])
    if vid is None:
        return None
    vid = str(vid).strip()
    options = extract_options(rec, ad["opts"])
    letter, aidx = normalize_answer(_first(rec, ad["ans"]), options, ad["base"])
    qid_raw = _first(rec, ad["qid"])
    query_id = str(qid_raw) if qid_raw is not None else f"{bench}-{idx:06d}"
    row = dict(
        video_id=vid,
        query_id=query_id,
        feature_ref=f"{feature_subdir}/{vid}.npy",
        prompt=build_prompt(_first(rec, ad["q"], ""), options),
        answer=letter,
        answer_index=aidx,
        options=options,
        task_type=str(_first(rec, ad["task"], bench)),
        bench=bench,
        split=str(rec.get("split", "test")),
    )
    bucket = _first(rec, ad["bucket"]) if ad["bucket"] else None
    if bucket is not None:
        row["duration_bucket"] = str(bucket).lower()
    # 透传视频路径（若源标注带 video_path，供 extract_features 直接定位视频，无需另建映射）
    vpath = _first(rec, ["video_path", "videoPath"])
    if vpath is not None:
        row["video_path"] = str(vpath)
    if "duration" in rec:
        try:
            row["duration"] = round(float(rec["duration"]), 3)
        except (TypeError, ValueError):
            pass
    return row


def run(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n, n_skip, n_noans = 0, 0, 0
    from collections import Counter
    buckets, tasks = Counter(), Counter()
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, rec in enumerate(_iter_records(args.src)):
            row = convert_record(rec, args.bench, i, args.feature_subdir)
            if row is None:
                n_skip += 1; continue
            if row["answer_index"] < 0:
                n_noans += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            tasks[row["task_type"]] += 1
            if "duration_bucket" in row:
                buckets[row["duration_bucket"]] += 1
    print(f"[convert:{args.bench}] 写入 {n} 行 → {args.out}（跳过无vid {n_skip}；无答案 {n_noans}）")
    if buckets:
        print(f"[convert:{args.bench}] duration 分档: {dict(buckets)}（VideoMME long 子集用于 GATE-4）")
    print(f"[convert:{args.bench}] 任务类分布(前5): {dict(tasks.most_common(5))}")
    print(f"[convert:{args.bench}] 下一步：extract_features 抽特征 + qa_eval 评准确率")


def main():
    ap = argparse.ArgumentParser(description="四基准 → 统一 QA manifest（真实适配器）")
    ap.add_argument("--bench", required=True, choices=list(ADAPTERS))
    ap.add_argument("--src", required=True, help="官方标注 json/jsonl（parquet 先转 jsonl）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature-subdir", default="features_npy", help="feature_ref 前缀")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
