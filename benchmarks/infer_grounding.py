#!/usr/bin/env python3
"""
benchmarks/infer_grounding.py — 连续查询定位推理：推理 manifest → pred.jsonl（闭合 CSG/VFR 环）。

闭环位置：csg_build/vfr_build → materialize（真实视频重采样特征）→ **本脚本** → csg_eval/vfr_eval。

机制（设计方案 §4.3 / 命题 2）：
  1) 读 materialize 产出的特征 [L,P,d] + 时间戳 [L]（可非均匀）；
  2) 在**比输入帧更密的亚帧网格** t_query 上，用 ContinuousQuery 在帧与帧之间任意时刻求读出 y(t*)；
  3) 对文本查询算相关度 score(t*)；
  4) **边界由阈值穿越点的线性插值求到亚帧精度**（boundaries_from_scores，纯函数）——
     这一步不吸附到网格，是穿透离散 δ/4 地板的关键，且**与模型无关、可精确单测**。

诚实边界：默认打分器 `HashDirectionScorer` 是**未训练占位**（query 条件但方向随机），只用于跑通链路/
对齐 pred 格式；真实数值需注入训练好的对齐/定位头（--scorer 或 API 注入）。真正体现"穿透 δ/4"的
亚帧边界提取逻辑在 boundaries_from_scores，由 tests/regression_infer_grounding.py 精确验证。

用法：
  # 跑通链路（本机 CPU，未训练模型，仅验证 plumbing + pred 格式）
  python infer_grounding.py --manifest csg_infer.jsonl --data-root . --out pred.jsonl --query-factor 8
  # 真实：注入训练权重
  python infer_grounding.py --manifest csg_infer.jsonl --data-root . --ckpt ckpt_final.pt --out pred.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from common import read_jsonl, write_jsonl             # noqa: E402


# ============================ 纯函数核心（穿透 δ/4 的关键，可精确单测）============================
def subframe_grid(ts, factor: int) -> np.ndarray:
    """在 [ts0, ts_{-1}] 上布置比输入帧密 factor 倍的查询网格（点落在帧间 → 亚帧）。"""
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) < 2:
        return ts.astype(np.float64)
    return np.linspace(float(ts[0]), float(ts[-1]), (len(ts) - 1) * int(factor) + 1)


def boundaries_from_scores(t_query, scores, rel_thresh: float = 0.5):
    """相关度曲线 score(t*) → 定位边界 (start,end)，边界用**阈值穿越线性插值**到亚帧精度。

    取峰值所在的连通高相关区间；左/右边界在"跨过阈值"的相邻两点间线性插值求解——
    因此边界取值于 ℝ、落在输入帧之间，突破离散模型只能落在网格上的 δ/4 下界（命题 2）。
    """
    t = np.asarray(t_query, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    Q = len(s)
    if Q == 0:
        return 0.0, 0.0
    if Q == 1:
        return float(t[0]), float(t[0])
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:                                 # 平坦 → 无定位信息，返回全区间
        return float(t[0]), float(t[-1])
    thr = lo + rel_thresh * (hi - lo)
    peak = int(np.argmax(s))
    i = peak
    while i - 1 >= 0 and s[i - 1] >= thr:
        i -= 1
    j = peak
    while j + 1 < Q and s[j + 1] >= thr:
        j += 1

    def cross(a: int, b: int) -> float:
        """在索引 a,b 之间线性插值出 score==thr 的时刻（a 在内 b 在外或反之皆可）。"""
        sa, sb = s[a], s[b]
        if abs(sb - sa) < 1e-12:
            return float(t[a])
        w = (thr - sa) / (sb - sa)
        w = min(max(w, 0.0), 1.0)
        return float(t[a] + w * (t[b] - t[a]))

    start = cross(i - 1, i) if i > 0 else float(t[i])   # 左：外(i-1)→内(i)
    end = cross(j, j + 1) if j < Q - 1 else float(t[j])  # 右：内(j)→外(j+1)
    if end < start:
        start, end = end, start
    return start, end


# ============================ 打分器（模型相关，可注入）============================
def hash_unit_vector(s: str, d: int, seed: int = 0) -> np.ndarray:
    """由查询串确定性地生成 R^d 单位向量（未训练占位方向）。"""
    h = int(hashlib.sha1(f"{seed}:{s}".encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(h).standard_normal(d)
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


class HashDirectionScorer:
    """默认打分器：真·CST-SSM 连续读出 y(t*) 与"查询哈希方向"的点积。

    **未训练占位**——方向随机，分数无语义，仅供跑通链路/对齐 pred 格式。生产替换为训练好的
    query 对齐头：实现 score(feats, ts, t_query, query)->(Q,) 即可注入。
    """

    def __init__(self, ckpt: str | None = None, d_model: int = 96, device: str = "cpu"):
        self.ckpt = ckpt
        self.d_model = d_model
        self.device = device
        self._model = None
        self._feat_dim = None

    def _build(self, feat_dim: int):
        import torch
        from cst_ssm.models import CSTSSMModel, CSTSSMConfig
        from cst_ssm.modules.llm_interface import LLMConfig
        cfg = CSTSSMConfig(input_mode="feature", feat_dim=feat_dim, d_model=self.d_model,
                           llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=64))
        model = CSTSSMModel(cfg).to(self.device).eval()
        if self.ckpt:
            from cst_ssm.utils import load_checkpoint      # 打印 missing/unexpected，不静默
            load_checkpoint(model, self.ckpt, tag="hash-scorer")
        self._model, self._feat_dim = model, feat_dim

    def score(self, feats: np.ndarray, ts: np.ndarray, t_query: np.ndarray, query: str) -> np.ndarray:
        import torch
        from cst_ssm.modules.continuous_query import ContinuousQuery
        if self._model is None or self._feat_dim != feats.shape[-1]:
            self._build(int(feats.shape[-1]))
        fe = torch.from_numpy(np.ascontiguousarray(feats)).float().unsqueeze(0).to(self.device)
        tt = torch.from_numpy(np.ascontiguousarray(ts)).float().unsqueeze(0).to(self.device)
        tq = torch.from_numpy(np.ascontiguousarray(t_query)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            x = self._model.vision(fe)                              # (1,L,d)
            cq = ContinuousQuery(self._model.temporal.branches[0])
            y = cq.query(x, tt, tq)[0].cpu().numpy()                # (Q,d)
        q = hash_unit_vector(query, y.shape[-1])
        return y @ q                                                # (Q,)


class TrainedGroundingScorer:
    """**训练版打分头**：从 --config 建 CST-SSM(含 GroundingHead) 并载 --ckpt，用 query 条件头打分。

    替换 HashDirectionScorer 的随机方向为真训练权重（由 scripts/train_grounding.py 产出）。
    未给 --ckpt 时头为随机初始化（会告警，仅验链路）；给 --ckpt 即接上真实权重。
    """

    def __init__(self, ckpt=None, config=None, d_model=96, device="cpu"):
        self.ckpt, self.config, self.d_model, self.device = ckpt, config, d_model, device
        self._model = None
        self._tok = None
        self._feat_dim = None

    def _build(self, feat_dim: int):
        import torch
        from cst_ssm.models import CSTSSMModel, CSTSSMConfig
        from cst_ssm.modules.llm_interface import LLMConfig
        from cst_ssm.data.dataset import ByteTokenizer
        if self.config:                                             # 用训练同款 config 对齐架构
            from cst_ssm.utils import model_config_from_dict, load_yaml
            d = load_yaml(self.config)["model"]; d["grounding_head"] = True
            cfg = model_config_from_dict(d)
        else:
            cfg = CSTSSMConfig(input_mode="feature", feat_dim=feat_dim, d_model=self.d_model,
                               grounding_head=True,
                               llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=64))
        model = CSTSSMModel(cfg).to(self.device).eval()
        if self.ckpt:
            from cst_ssm.utils import load_checkpoint      # 打印 missing/unexpected，不静默
            load_checkpoint(model, self.ckpt, tag="grounding-scorer")
        self._model, self._tok, self._feat_dim = model, ByteTokenizer(), feat_dim

    def score(self, feats, ts, t_query, query: str) -> np.ndarray:
        import torch
        if self._model is None or self._feat_dim != feats.shape[-1]:
            self._build(int(feats.shape[-1]))
        ids = self._tok.encode(query)[: 512]
        fe = torch.from_numpy(np.ascontiguousarray(feats)).float().unsqueeze(0).to(self.device)
        tt = torch.from_numpy(np.ascontiguousarray(ts)).float().unsqueeze(0).to(self.device)
        tq = torch.from_numpy(np.ascontiguousarray(t_query)).float().unsqueeze(0).to(self.device)
        iid = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        am = torch.ones_like(iid, dtype=torch.bool)
        with torch.no_grad():
            logits = self._model.ground_scores(fe, tt, iid, tq, am)[0].cpu().numpy()   # (Q,)
        return logits


def build_scorer(args):
    """按 --scorer 选打分器：trained（训练头，默认）| hash（随机占位，仅链路自测）。"""
    kind = args.scorer
    if kind == "hash":
        return HashDirectionScorer(ckpt=args.ckpt, d_model=args.d_model, device=args.device)
    return TrainedGroundingScorer(ckpt=args.ckpt, config=args.config,
                                  d_model=args.d_model, device=args.device)


# ============================ 特征加载 + 批推理 ============================
def load_features(feature_ref: str, data_root: str):
    """兼容 .npz 与 .npy(+.ts.npy)，与 VideoTemporalDataset 一致。"""
    path = os.path.join(data_root, feature_ref)
    if feature_ref.endswith(".npy"):
        feats = np.load(path, mmap_mode="r")
        ts = np.load(path[:-4] + ".ts.npy")
    else:
        z = np.load(path)
        feats, ts = z["features"], z["timestamps"]
    return np.asarray(feats, np.float32), np.asarray(ts, np.float32)


def run(args):
    rows = read_jsonl(args.manifest)
    scorer = build_scorer(args)
    out, n_fail = [], 0
    fails = []
    for r in rows:
        try:
            feats, ts = load_features(r["feature_ref"], args.data_root)
            tq = subframe_grid(ts, args.query_factor)
            scores = scorer.score(feats, ts, tq, r.get("prompt", r.get("query", "")))
            ps, pe = boundaries_from_scores(tq, scores, args.rel_thresh)
            out.append({"query_id": r["query_id"], "pred_start": round(ps, 4),
                        "pred_end": round(pe, 4)})
        except Exception as e:
            n_fail += 1
            fails.append({"query_id": r.get("query_id"), "error": str(e)})
    write_jsonl(args.out, out)
    if fails:
        write_jsonl(args.out + ".failed", fails)
    print(f"[infer] 出预测 {len(out)} 条 → {args.out}{'；失败 %d' % n_fail if n_fail else ''}")
    if not args.ckpt:
        head = "HashDirection(随机方向)" if args.scorer == "hash" else "GroundingHead(随机初始化)"
        print(f"[infer] ⚠ 未加载权重：{head} 未训练，pred 无语义（仅验证链路/格式）。"
              "\n[infer]   真实评测请 --ckpt 注入 train_grounding 产出的权重；亚帧机制见 boundaries_from_scores。")
    print(f"[infer] 评测：python csg_eval.py --manifest <原csg_*.jsonl> --pred {args.out}")


def main():
    ap = argparse.ArgumentParser(description="连续查询定位推理 → pred.jsonl")
    ap.add_argument("--manifest", required=True, help="materialize 产出的 *_infer.jsonl")
    ap.add_argument("--data-root", default=".", help="feature_ref 的根")
    ap.add_argument("--out", required=True, help="预测 jsonl {query_id,pred_start,pred_end}")
    ap.add_argument("--ckpt", default=None, help="train_grounding 产出的权重（文件或分片目录）；缺省未训练")
    ap.add_argument("--config", default=None, help="与训练同款 config（对齐架构以正确加载 --ckpt）")
    ap.add_argument("--scorer", default="trained", choices=["trained", "hash"],
                    help="trained=GroundingHead 训练头(默认)；hash=随机方向占位(仅链路自测)")
    ap.add_argument("--query-factor", type=int, default=8, help="亚帧网格相对输入帧的加密倍数")
    ap.add_argument("--rel-thresh", type=float, default=0.5, help="相关度阈值（相对 min~max）")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
