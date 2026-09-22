"""Convert benchmark annotations to a unified QA manifest format."""
from __future__ import annotations

import argparse
import json
import os
import re
import string

from .dist_utils import feature_stem

LETTERS = string.ascii_uppercase
_OPT_PREFIX = re.compile(r"^\(?([A-Z])[.)]\s+", re.I)


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
    """Extract option texts, stripping existing letter prefixes."""
    raw = _first(rec, opt_keys)
    if raw is None:
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
    if isinstance(raw, dict):
        raw = [raw[k] for k in sorted(raw)]
    if not isinstance(raw, list):
        raw = [raw]
    out = []
    for o in raw:
        s = str(o).strip()
        out.append(_OPT_PREFIX.sub("", s).strip())
    return out


def normalize_answer(raw, options: list, base: int = 1):
    """Normalize an answer to (letter, 0-based index); ("", -1) means unknown."""
    n = len(options)
    if raw is None or raw == "":
        return "", -1
    s = str(raw).strip()
    if len(s) == 1 and s.upper() in LETTERS:
        idx = LETTERS.index(s.upper())
        if n == 0 or idx < n:
            return LETTERS[idx], idx
    if s.lstrip("-").isdigit():
        v = int(s)
        idx = v - base if base else v
        if 0 <= idx < max(n, 1):
            return LETTERS[idx], idx
        alt = v - (1 - base)
        if 0 <= alt < max(n, 1):
            return LETTERS[alt], alt
        return "", -1
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


def convert_record(rec: dict, bench: str, idx: int, feature_subdir: str,
                   feature_ext: str = ".npy") -> dict | None:
    ad = ADAPTERS[bench]
    vid = _first(rec, ad["vid"])
    if vid is None:
        return None
    vid = feature_stem(vid)
    options = extract_options(rec, ad["opts"])
    letter, aidx = normalize_answer(_first(rec, ad["ans"]), options, ad["base"])
    qid_raw = _first(rec, ad["qid"])
    query_id = str(qid_raw) if qid_raw is not None else f"{bench}-{idx:06d}"
    row = dict(
        video_id=vid,
        query_id=query_id,
        feature_ref=f"{feature_subdir}/{vid}{feature_ext}",
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
            row = convert_record(rec, args.bench, i, args.feature_subdir, args.feature_ext)
            if row is None:
                n_skip += 1; continue
            if row["answer_index"] < 0:
                n_noans += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            tasks[row["task_type"]] += 1
            if "duration_bucket" in row:
                buckets[row["duration_bucket"]] += 1
    print(f"[convert:{args.bench}] wrote {n} rows -> {args.out} "
          f"(skipped {n_skip} without video id; {n_noans} without answer)")
    if buckets:
        print(f"[convert:{args.bench}] duration buckets: {dict(buckets)}")
    print(f"[convert:{args.bench}] top tasks: {dict(tasks.most_common(5))}")


def main():
    ap = argparse.ArgumentParser(description="Benchmark annotations to unified QA manifest")
    ap.add_argument("--bench", required=True, choices=list(ADAPTERS))
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature-subdir", default="features_npy")
    ap.add_argument("--feature-ext", default=".npy", choices=[".npy", ".npz"])
    a = ap.parse_args()
    print(f"[convert:{a.bench}] feature_ref looks like {a.feature_subdir}/<video_id>{a.feature_ext}")
    run(a)


if __name__ == "__main__":
    main()
