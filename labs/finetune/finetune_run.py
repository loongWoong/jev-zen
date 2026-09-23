#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laya 微调框架 · 头训练器（head-only，纯 numpy，无 torch 依赖）

设计要点（详见 FRAMEWORK.md Phase 0 / §4）：
  - 编码器 22 层冻结；一次前向把所有样本的 marker_h（决策头输入）缓存下来，
    此后训练只在 scorer（LN→Linear→gelu→Linear，op 索引 0,1,3）上做解析梯度下降。
  - choice 问题的 4 个选项 type 都是 choice，type_emb 对其同加 → 不改排名，
    故头训练只动 scorer 即可改变 choice 的 top-1 排序。
  - 训练目标：每个样本的 choice logits 上，教师方向 rd.choice 的 CE。

用法：
  python finetune_run.py --dataset artifacts/maze_dataset.jsonl --epochs 80
  python finetune_run.py --dataset artifacts/maze_dataset.jsonl --out-model artifacts/model_maze_ft.safetensors
"""
from __future__ import annotations
import os
import sys
import json
import struct
import argparse
import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from laya_verify import build  # noqa: E402
from benchmark import eval_label_accuracy  # noqa: E402

DEFAULT_MODEL = os.path.join("C:/Users/Administrator/Downloads/jev-zen", "model.safetensors")
MOVE_ORDER = ["up", "down", "left", "right"]
EPS = 1e-5


# ---------------------------------------------------------------------------
# 解析 gelu（与 laya_verify.act_fn "gelu" 一致：A&S 7.1.26 近似 erf）
# ---------------------------------------------------------------------------
def gelu_fwd(x):
    a = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * a)
    erf = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                  - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    erf = np.where(x < 0, -erf, erf)
    return 0.5 * x * (1.0 + erf)


def gelu_bwd(x):
    # d/dx [0.5*x*(1+erf(x/sqrt2))] 的解析梯度（用真 erf 的导数，与近似前向误差 <1.5e-7）
    a = x / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(a))
    erf = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                  - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    erf = np.where(x < 0, -erf, erf)
    sig = np.exp(-a * a)
    d_erf = math.sqrt(2.0 / math.pi) * sig
    return 0.5 * (1.0 + erf) + 0.5 * x * d_erf


# ---------------------------------------------------------------------------
# scorer 前向 / 反向（仅 6 个参数：w0,b0,w1,b1,w3,b3）
# ---------------------------------------------------------------------------
def scorer_forward(markers, P):
    w0, b0, w1, b1, w3, b3 = P["w0"], P["b0"], P["w1"], P["b1"], P["w3"], P["b3"]
    cache = {}
    # op0: LayerNorm
    mu = markers.mean(-1, keepdims=True)
    var = markers.var(-1, keepdims=True)
    hn = (markers - mu) / np.sqrt(var + EPS)
    h = hn * w0 + b0
    cache["ln"] = (mu, var, hn, w0)
    # op1: Linear 768->768
    a1_in = h
    h = a1_in @ w1.T + b1
    cache["lin1"] = (a1_in, w1, b1)
    # gap: gelu
    gelu_x = h
    h = gelu_fwd(h)
    cache["gelu_x"] = gelu_x
    # op3: Linear 768->1
    a3_in = h
    logits = a3_in @ w3.T + b3
    cache["lin3"] = (a3_in, w3, b3)
    return logits.reshape(-1), cache


def scorer_backward(dlogits, cache):
    # dlogits: [M]
    # op3
    dA3 = dlogits[:, None]                                   # [M,1]
    a3_in, w3, b3 = cache["lin3"]
    dW3 = dA3.T @ a3_in                                      # [1,768]
    dB3 = dA3.sum(0)                                         # [1]
    dA3_in = dA3 @ w3                                        # [M,768]
    # gelu
    dGelu = dA3_in * gelu_bwd(cache["gelu_x"])               # [M,768]
    # op1
    a1_in, w1, b1 = cache["lin1"]
    dW1 = dGelu.T @ a1_in                                    # [768,768]
    dB1 = dGelu.sum(0)                                       # [768]
    dA1_in = dGelu @ w1                                      # [M,768]
    # op0 LayerNorm
    mu, var, hn, w0 = cache["ln"]
    dhn = dA1_in * w0                                        # [M,768]
    dW0 = (dA1_in * hn).sum(0)                               # [768]
    dB0 = dA1_in.sum(0)                                      # [768]
    return {"w0": dW0, "b0": dB0, "w1": dW1, "b1": dB1, "w3": dW3, "b3": dB3}


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------
class Adam:
    def __init__(self, params_keys, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = {k: np.zeros_like(params_keys[k]) for k in params_keys}
        self.v = {k: np.zeros_like(params_keys[k]) for k in params_keys}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in grads:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            grads[k] = self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return grads


# ---------------------------------------------------------------------------
# safetensors 写出（保留原始 dtype）
# ---------------------------------------------------------------------------
_DT = {"float16": "F16", "float32": "F32", "float64": "F64"}


def save_safetensors(path, tensors: dict):
    header = {}
    data = b""
    offset = 0
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        dt = _DT[a.dtype.name]
        nb = a.tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [offset, offset + len(nb)]}
        data += nb
        offset += len(nb)
    header["__metadata__"] = {"format": "pt"}
    hjson = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(data)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Laya 头微调（numpy）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default=os.path.join(HERE, "artifacts", "maze_dataset.jsonl"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.003,
                    help="微调学习率（原 0.02 在 148 样本上过拟合→通道门爆炸，降到 3e-3）")
    ap.add_argument("--wd", type=float, default=1e-4, help="scorer 权重衰减")
    ap.add_argument("--calib-penalty", type=float, default=0.0,
                    help="logit 极差硬惩罚系数（默认 0=关闭；校准改由 --label-smoothing 负责）")
    ap.add_argument("--spread-cap", type=float, default=3.0,
                    help="logit 极差上限（仅当 --calib-penalty>0 时生效）")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="CE 目标平滑（0.1 ⇒ 金 0.925/其余 0.025），天然压制过自信并兼作正则")
    ap.add_argument("--val-frac", type=float, default=0.2, help="留出验证集比例")
    ap.add_argument("--early-stop", action="store_true",
                    help="val_acc 连续未升时早停（并还原最佳）")
    ap.add_argument("--out-model", default=os.path.join(HERE, "artifacts", "model_maze_ft.safetensors"))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--log-every", type=int, default=10,
                    help="每 N 个 epoch 打印一次 loss（控制端用于绘制实时曲线）")
    ap.add_argument("--cache-markers", default=os.path.join(HERE, "artifacts", "maze_markers.npz"))
    args = ap.parse_args()

    st, tok, pred = build(args.model)
    model = pred.model

    # 1) 载入样本，预计算 marker_h（带缓存）
    samples = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if os.path.isfile(args.cache_markers):
        print(f"[ft] 复用缓存 markers: {args.cache_markers}")
        dat = np.load(args.cache_markers, allow_pickle=True)
        mkeys = sorted([f for f in dat.files if f.startswith("m")],
                       key=lambda x: int(x[1:]))
        markers_list = [dat[k] for k in mkeys]
        gold_idx = dat["gold"].tolist()
    else:
        markers_list = []
        gold_idx = []
        for i, smp in enumerate(samples):
            ids = smp["ids"]
            mp = smp["marker_pos"]
            mt = smp["marker_type"]
            out = model.forward(ids, mp, mt)
            mh = out["marker_h"]                      # [M,768]
            key = (smp.get("labels") or {}).get("move", {}).get("key")
            if key is None or key not in MOVE_ORDER:
                print(f"[warn] 跳过无 move 标签样本: {smp.get('step')}")
                continue
            markers_list.append(np.asarray(mh, dtype=np.float32))
            gold_idx.append(MOVE_ORDER.index(key))
            if (i + 1) % 30 == 0:
                print(f"[ft] 预计算 marker_h {i + 1}/{len(samples)}", flush=True)
        np.savez(args.cache_markers,
                 **{f"m{i}": m for i, m in enumerate(markers_list)},
                 gold=np.array(gold_idx, dtype=np.int64))
        print(f"[ft] markers 缓存 → {args.cache_markers}", flush=True)
    n = len(markers_list)
    print(f"[ft] 样本 {n}；marker_h 维度 {markers_list[0].shape}", flush=True)

    # 2) 初始化可训练 scorer 参数（fp32）
    P = {
        "w0": st.get("scorer.0.weight").astype(np.float32).copy(),
        "b0": st.get("scorer.0.bias").astype(np.float32).copy(),
        "w1": st.get("scorer.1.weight").astype(np.float32).copy(),
        "b1": st.get("scorer.1.bias").astype(np.float32).copy(),
        "w3": st.get("scorer.3.weight").astype(np.float32).copy(),
        "b3": st.get("scorer.3.bias").astype(np.float32).copy(),
    }

    # 3) 准确率（直接用冻结的 marker_h + scorer 计算，等价于模型的 choice top-1，
    #    因为 logits = scorer(marker_h) 且 marker_h 与 scorer 无关）
    def compute_acc(Pp):
        hit = 0
        for mi, gi in zip(markers_list, gold_idx):
            logits, _ = scorer_forward(mi, Pp)
            # 仅取 move 问题（问题 0）的 4 个选项 logits 判定
            if int(np.argmax(logits[:4])) == gi:
                hit += 1
        return hit / n

    before_acc = compute_acc(P)

    # 4) 训练（带校准围栏：spread 惩罚 + 权重衰减 + 留出验证集早停）
    # 切分训练/验证（按索引等距切分，确定性、可复现）
    n_val = max(1, int(round(n * args.val_frac)))
    val_idx = set(range(0, n, max(1, n // n_val)))
    train_markers = [markers_list[i] for i in range(n) if i not in val_idx]
    train_gold = [gold_idx[i] for i in range(n) if i not in val_idx]
    val_markers = [markers_list[i] for i in val_idx]
    val_gold = [gold_idx[i] for i in val_idx]
    print(f"[ft] 训练样本 {len(train_markers)} / 验证样本 {len(val_markers)}", flush=True)

    FAIL_SPREAD = 5.0  # 放宽：校准改由 label smoothing 负责，这里仅防极端爆炸
    opt = Adam(P, lr=args.lr)
    best_P = None
    best_val = -1.0
    patience = 0
    for epoch in range(args.epochs):
        tot_loss = 0.0
        grads = {k: np.zeros_like(P[k]) for k in P}
        for mi, gi in zip(train_markers, train_gold):
            logits, cache = scorer_forward(mi, P)
            mv = logits[:4]                         # move 的 4 个选项（问题 0）
            # 稳定 softmax + CE（带 label smoothing 做校准/正则）
            z = mv - mv.max()
            e = np.exp(z)
            p = e / e.sum()
            K = mv.shape[0]
            eps = args.label_smoothing
            tgt = np.full(K, eps / K)
            tgt[gi] = 1.0 - eps * (K - 1) / K
            tot_loss += float(-np.sum(tgt * np.log(p + 1e-12)))
            dlogits_mv = p - tgt                       # 平滑 CE 对 logits 的梯度
            # 可选硬围栏：压制 logit 极差（默认关闭，靠 label smoothing 即可）
            spread = float(mv.max() - mv.min())
            if args.calib_penalty > 0 and spread > args.spread_cap:
                dlogits_mv[int(np.argmax(mv))] += args.calib_penalty
                dlogits_mv[int(np.argmin(mv))] -= args.calib_penalty
            # 组装完整梯度（scorer 输出 8 个 logit：move[0:4] 监督，closeness[4:8] 不监督）
            dlogits_full = np.zeros_like(logits)
            dlogits_full[:4] = dlogits_mv
            g = scorer_backward(dlogits_full, cache)
            for k in grads:
                grads[k] += g[k]
        # 平均梯度 + Adam 更新（+ 权重衰减）
        for k in grads:
            grads[k] /= len(train_markers)
        upd = opt.step(grads)
        for k in P:
            P[k] = P[k] - upd[k] - args.lr * args.wd * P[k]
        # 验证
        v_loss = 0.0
        v_hit = 0
        v_spread = 0.0
        for mi, gi in zip(val_markers, val_gold):
            logits, _ = scorer_forward(mi, P)
            mv = logits[:4]
            z = mv - mv.max(); e = np.exp(z); p = e / e.sum()
            v_loss += float(-np.log(p[gi] + 1e-12))
            v_spread = max(v_spread, float(mv.max() - mv.min()))
            if int(np.argmax(mv)) == gi:
                v_hit += 1
        v_loss /= max(1, len(val_markers))
        v_acc = v_hit / max(1, len(val_markers))
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"[ft] epoch {epoch + 1:3d}/{args.epochs}  loss={tot_loss / len(train_markers):.4f}"
                  f"  val_loss={v_loss:.4f}  val_acc={v_acc:.3f}  val_spread={v_spread:.3f}", flush=True)
        # 早停：校准崩了直接还原最佳并退出
        if v_spread > FAIL_SPREAD:
            print(f"[ft] epoch {epoch + 1}: val_spread={v_spread:.3f} > {FAIL_SPREAD} → 早停（还原最佳）", flush=True)
            if best_P is not None:
                P = {k: best_P[k].copy() for k in best_P}
            break
        if v_acc > best_val:
            best_val = v_acc
            best_P = {k: P[k].copy() for k in P}
            patience = 0
        else:
            patience += 1
            if args.early_stop and patience >= 8:
                print(f"[ft] epoch {epoch + 1}: val_acc 连续 {patience} 轮未升 → 早停（还原最佳）", flush=True)
                P = {k: best_P[k].copy() for k in best_P}
                break
    else:
        # 正常跑完：若曾出现更好模型则还原
        if best_P is not None:
            P = {k: best_P[k].copy() for k in best_P}

    after_acc = compute_acc(P)

    # 5) 把训练后的 scorer 写回 st 缓存（fp32），供按需核对
    for name, key in [("scorer.0.weight", "w0"), ("scorer.0.bias", "b0"),
                      ("scorer.1.weight", "w1"), ("scorer.1.bias", "b1"),
                      ("scorer.3.weight", "w3"), ("scorer.3.bias", "b3")]:
        st._cache[(name, True)] = P[key].astype(np.float32)

    # 6) 写出微调模型（保留原始 dtype）
    if not args.no_save:
        try:
            tensors = {}
            for nm in st.keys:
                tensors[nm] = st.raw(nm)              # 原始 dtype
            for name, key in [("scorer.0.weight", "w0"), ("scorer.0.bias", "b0"),
                              ("scorer.1.weight", "w1"), ("scorer.1.bias", "b1"),
                              ("scorer.3.weight", "w3"), ("scorer.3.bias", "b3")]:
                tensors[name] = P[key].astype(tensors[name].dtype)
            save_safetensors(args.out_model, tensors)
            print(f"[ft] 微调模型写出 → {args.out_model}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ft] 模型写出失败（不影响训练结果）：{e}", flush=True)

    print("\n=== 微调结果（迷宫数据集 choice top-1，n=%d）===" % n, flush=True)
    print(f"训练前: {before_acc:.4f}", flush=True)
    print(f"训练后: {after_acc:.4f}", flush=True)
    print(f"Δ     : {round(after_acc - before_acc, 4)}", flush=True)
    if before_acc < 1e-9:
        print("注：训练前为 0 → 印证 Phase-0「通道未对齐（WARN）」：原始 scorer 在迷宫 choice 上无判别力。", flush=True)
    if after_acc - before_acc > 0.05:
        print("结论：头训练（scorer）即可把 choice top-1 拉高 → 选项 marker 向量存在可被 scorer 利用的区分信号。", flush=True)
    else:
        print("结论：Δ 较小 → 即便训练 scorer，选项 marker 向量区分度仍弱；须先解 Phase-0 通道对齐再谈微调。", flush=True)


if __name__ == "__main__":
    main()
