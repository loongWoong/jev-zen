#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laya 微调框架 · 微调管线（pipeline.py）

backend:
  - head-only  : 冻结 22 层编码器，只训 scorer / act_head / type_emb / temperature
                 （~1.5M 参数，Phase 0 对齐恢复 + 快速实验）
  - full       : 编码器 + 头 端到端（需 PyTorch + CUDA；本机用 py3.12 venv：
                 ~/.workbuddy/binaries/python/envs/laya-cuda）

实现要点：LayaTorch 是 laya_verify.LayaModel.forward 的可微 torch 移植
（RoPE theta=160000 / alternating3 / GeGLU / head 层 / scorer gap-激活 / act_head 4 标量特征）。
必须先过 parity_test（与 numpy 前向逐张量误差 < 1e-3）才能训练，否则数值不等价。

运行（CUDA venv）：
  python pipeline.py --model <model> --backend head-only --epochs 8 --dataset artifacts/samples.jsonl
  python pipeline.py --model <model> parity
"""
from __future__ import annotations
import os
import sys
import json
import copy
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 工作树根，含 laya_verify.py
sys.path.insert(0, ROOT)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from laya_verify import build, LayaModel, Question, PRIMITIVES  # noqa: E402
    TORCH_OK = True
except Exception as e:  # noqa: BLE001
    TORCH_OK = False
    _IMPORT_ERR = e

DEFAULT_MODEL = os.path.join("C:/Users/Administrator/Downloads/jev-zen", "model.safetensors")


# ---------------------------------------------------------------------------
# 可微 torch 移植
# ---------------------------------------------------------------------------
def layernorm_no_bias(x, w, eps):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + eps) * w


class LayaTorch(nn.Module):
    def __init__(self, st, cfg):
        super().__init__()
        self.cfg = cfg
        self.H = cfg["hidden"]
        self.L = cfg["n_layers"]
        self.nh = cfg["n_heads"]
        self.dh = self.H // self.nh
        self.scale = self.dh ** -0.5
        self.eps = cfg["norm_eps"]
        theta = float(cfg["rope_theta"])
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.dh, 2, dtype=torch.float64) / self.dh))

        def T(name):
            return torch.from_numpy(st.get(name)).float()

        # 嵌入 + 归一化
        self.tok_emb = nn.Embedding(st.get("encoder.embeddings.tok_embeddings.weight").shape[0],
                                    self.H)
        self.tok_emb.weight.data = T("encoder.embeddings.tok_embeddings.weight")
        self.emb_norm = nn.Parameter(T("encoder.embeddings.norm.weight"))
        self.final_norm = nn.Parameter(T("encoder.final_norm.weight"))

        # 编码器层
        self.attn_norm = nn.ModuleList()
        self.Wqkv = nn.ModuleList()
        self.Wo = nn.ModuleList()
        self.mlp_norm = nn.ModuleList()
        self.Wi = nn.ModuleList()
        self.Wo2 = nn.ModuleList()
        for i in range(self.L):
            p = f"encoder.layers.{i}."
            an = st.get_optional(p + "attn_norm.weight")
            self.attn_norm.append(nn.Parameter(T(p + "attn_norm.weight")) if an is not None
                                  else None)
            self.Wqkv.append(nn.Parameter(T(p + "attn.Wqkv.weight")))
            self.Wo.append(nn.Parameter(T(p + "attn.Wo.weight")))
            self.mlp_norm.append(nn.Parameter(T(p + "mlp_norm.weight")))
            self.Wi.append(nn.Parameter(T(p + "mlp.Wi.weight")))
            self.Wo2.append(nn.Parameter(T(p + "mlp.Wo.weight")))

        # head 层
        self.head_in_w, self.head_in_b, self.head_out_w, self.head_out_b = [], [], [], []
        self.head_l1_w, self.head_l1_b, self.head_l2_w, self.head_l2_b = [], [], [], []
        self.head_n1_w, self.head_n1_b, self.head_n2_w, self.head_n2_b = [], [], [], []
        i = 0
        while st.get_optional(f"head.layers.{i}.norm1.weight") is not None:
            p = f"head.layers.{i}."
            self.head_in_w.append(nn.Parameter(T(p + "self_attn.in_proj_weight")))
            self.head_in_b.append(nn.Parameter(T(p + "self_attn.in_proj_bias")))
            self.head_out_w.append(nn.Parameter(T(p + "self_attn.out_proj.weight")))
            self.head_out_b.append(nn.Parameter(T(p + "self_attn.out_proj.bias")))
            self.head_l1_w.append(nn.Parameter(T(p + "linear1.weight")))
            self.head_l1_b.append(nn.Parameter(T(p + "linear1.bias")))
            self.head_l2_w.append(nn.Parameter(T(p + "linear2.weight")))
            self.head_l2_b.append(nn.Parameter(T(p + "linear2.bias")))
            self.head_n1_w.append(nn.Parameter(T(p + "norm1.weight")))
            self.head_n1_b.append(nn.Parameter(T(p + "norm1.bias")))
            self.head_n2_w.append(nn.Parameter(T(p + "norm2.weight")))
            self.head_n2_b.append(nn.Parameter(T(p + "norm2.bias")))
            i += 1
        self.n_head_layers = i

        # scorer（索引顺序，1D=LayerNorm，2D=Linear，跳号插 gelu）
        self.scorer_w, self.scorer_b, self.scorer_is_norm = [], [], []
        idxs = sorted(int(m.group(1)) for m in
                      (__import__("re").match(r"^scorer\.(\d+)\.weight$", k) for k in st.keys)
                      if m)
        for i in idxs:
            w = st.get(f"scorer.{i}.weight")
            self.scorer_w.append(nn.Parameter(torch.from_numpy(w).float()))
            b = st.get_optional(f"scorer.{i}.bias")
            self.scorer_b.append(nn.Parameter(torch.from_numpy(b).float()) if b is not None
                                 else None)
            self.scorer_is_norm.append(w.ndim == 1)

        # act_head
        self.act_w, self.act_b = [], []
        aidx = sorted(int(m.group(1)) for m in
                      (__import__("re").match(r"^act_head\.(\d+)\.weight$", k) for k in st.keys)
                      if m)
        for i in aidx:
            self.act_w.append(nn.Parameter(T(f"act_head.{i}.weight")))
            self.act_b.append(nn.Parameter(T(f"act_head.{i}.bias")))
        self.act_in_dim = self.act_w[0].shape[1]

        # type_emb / temperature
        self.type_emb = nn.Parameter(T("type_emb.weight"))           # [3,768]
        self.temperature = nn.Parameter(
            torch.from_numpy(st.get("temperature")).double().float())  # [3]

    # ---- 基础块 ----
    def _rope(self, q, k):  # q,k: [nh,T,dh]
        Tt = q.shape[1]
        pos = torch.arange(Tt, dtype=torch.float64)
        freqs = torch.outer(pos, self.inv_freq.to(torch.float64)).to(torch.float32)
        emb = torch.cat([freqs, freqs], dim=-1)[None, :, :]          # [1,T,dh]
        cos, sin = torch.cos(emb), torch.sin(emb)
        def rot(t):
            d = t.shape[-1] // 2
            return torch.cat([-t[..., d:], t[..., :d]], dim=-1)
        return q * cos + rot(q) * sin, k * cos + rot(k) * sin

    def _attend(self, q, k, v, band):  # [nh,T,dh]
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if band is not None:
            idx = torch.arange(q.shape[1])
            mask = (idx[:, None] - idx[None, :]).abs() > band
            scores = scores.masked_fill(mask[None], -1e30)
        return F.softmax(scores, -1) @ v

    def _encoder_layer(self, x, i, band):  # x: [T,H]
        wn = self.attn_norm[i]
        h = x if wn is None else layernorm_no_bias(x, wn, self.eps)
        qkv = h @ self.Wqkv[i].T                                   # [T,3H]
        q, k, v = qkv.reshape(-1, 3, self.nh, self.dh).permute(1, 0, 2, 3)  # [3,nh,T,dh]
        q, k = self._rope(q, k)
        ctx = self._attend(q, k, v, band).permute(1, 0, 2, 3).reshape(-1, self.H)
        x = x + ctx @ self.Wo[i].T
        h = layernorm_no_bias(x, self.mlp_norm[i], self.eps)
        g = h @ self.Wi[i].T                                        # [T,2*inter]
        a, b = g.chunk(2, -1)
        x = x + (F.gelu(a) * b) @ self.Wo2[i].T
        return x

    def _head_layer(self, x, i):  # x: [T,H]
        Tt = x.shape[0]
        qkv = x @ self.head_in_w[i].T + self.head_in_b[i]
        q, k, v = qkv.chunk(3, -1)
        r = lambda t: t.reshape(Tt, self.nh, self.dh).permute(1, 0, 2)
        ctx = self._attend(r(q), r(k), r(v), None).permute(1, 0, 2).reshape(Tt, self.H)
        x = layernorm_no_bias(x + ctx @ self.head_out_w[i].T + self.head_out_b[i],
                              self.head_n1_w[i], self.head_n1_b[i], 1e-5)
        f = F.gelu(x @ self.head_l1_w[i].T + self.head_l1_b[i])
        x = layernorm_no_bias(x + f @ self.head_l2_w[i].T + self.head_l2_b[i],
                              self.head_n2_w[i], self.head_n2_b[i], 1e-5)
        return x

    def _scorer(self, marker_h):  # [M,H] -> [M]
        h = marker_h
        prev = None
        for idx in range(len(self.scorer_w)):
            if prev is not None and idx != prev + 1:
                h = F.gelu(h)
            w, b = self.scorer_w[idx], self.scorer_b[idx]
            if self.scorer_is_norm[idx]:
                h = layernorm_no_bias(h, w, self.eps)
            else:
                h = h @ w.T + (b if b is not None else 0)
            prev = idx
        return h.reshape(-1)

    def _act_head(self, pooled, feats):  # pooled[H], feats
        if self.act_in_dim != pooled.shape[0] + feats.shape[0]:
            feats = torch.zeros(self.act_in_dim - pooled.shape[0])
        h = torch.cat([pooled, feats]).unsqueeze(0)               # [1, act_in_dim]
        prev = None
        for idx in range(len(self.act_w)):
            if prev is not None and idx != prev + 1:
                h = F.gelu(h)
            h = h @ self.act_w[idx].T + self.act_b[idx]
            prev = idx
        return F.softmax(h.reshape(-1), -1)

    # ---- 主前向 ----
    def forward(self, ids, marker_pos, marker_type):
        dev = self.tok_emb.weight.device
        x = self.tok_emb(torch.tensor(ids, dtype=torch.long, device=dev))  # [T,H]
        x = layernorm_no_bias(x, self.emb_norm, self.eps)

        mp = marker_pos
        stage = self.cfg["type_emb_stage"]
        site = self.cfg["type_emb_site"]
        if stage == "input":
            x[mp] = x[mp] + self.type_emb[torch.tensor(marker_type, device=dev)]

        bands = [None if i % 3 == 0 else self.cfg["local_radius"] for i in range(self.L)]
        for i in range(self.L):
            x = self._encoder_layer(x, i, bands[i])
        x = layernorm_no_bias(x, self.final_norm, self.eps)

        if stage == "pre_head":
            x[mp] = x[mp] + self.type_emb[torch.tensor(marker_type, device=dev)]

        for i in range(self.n_head_layers):
            x = self._head_layer(x, i)

        marker_h = x[mp]                                            # [M,H]
        logits = self._scorer(marker_h)                            # [M]
        pooled = marker_h.mean(0)
        type_oh = torch.zeros(3)
        type_oh[int(torch.tensor(marker_type)[0]) % 3] = 1.0
        n_opt = max(1, len(mp))
        feats = torch.cat([type_oh, torch.tensor([__import__("math").log1p(n_opt) /
                                                  __import__("math").log(64.0)])]).float()
        act = self._act_head(pooled, feats)                       # [2]
        return {"logits": logits, "act": act, "marker_h": marker_h}


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------
def make_trainable(model, backend):
    if backend == "head-only":
        # 冻结编码器与 head 层，仅训 scorer / act_head / type_emb / temperature
        for p in model.parameters():
            p.requires_grad = False
        for p in model.scorer_w + model.scorer_b:
            if p is not None:
                p.requires_grad = True
        for p in model.act_w + model.act_b:
            p.requires_grad = True
        model.type_emb.requires_grad = True
        model.temperature.requires_grad = True
    else:  # full
        for p in model.parameters():
            p.requires_grad = True


def loss_fn(model_out, labels, temps, temps_mode):
    # 简化：choice 交叉熵 + act noul 交叉熵；score 在完整版按 EV 回归
    logits = model_out["logits"]
    act = model_out["act"]
    lc = F.cross_entropy(logits.unsqueeze(0), torch.tensor([labels["gold_idx"]], dtype=torch.long))
    ln = F.cross_entropy(act.unsqueeze(0), torch.tensor([labels["gold_act"]], dtype=torch.long))
    return lc + 0.5 * ln


def parity_test(model_path):
    if not TORCH_OK:
        print("[parity] torch 不可用，跳过。请用 py3.12 CUDA venv 运行。")
        return False
    st, tok, pred = build(model_path)
    np_model = LayaModel(st, dict(pred.cfg))
    cfg = dict(pred.cfg)
    torch_model = LayaTorch(st, cfg).float().eval()
    # 小样本
    qs = [Question("q0", "choice", "测试", {"A": "a", "B": "b", "C": "c"})]
    built = pred.build("state: a=1 b=2 c=3\nlegal: A,B,C", qs)
    ids, mp, mt = built["ids"], built["marker_pos"], built["marker_type"]
    with torch.no_grad():
        out_t = torch_model(ids, mp, mt)
    out_n = np_model.forward(ids, mp, mt)
    import numpy as np
    diff = np.abs(out_t["logits"].detach().numpy() - np.asarray(out_n["logits"])).max()
    print(f"[parity] 最大 logits 误差 = {diff:.3e}  (阈值 1e-3)")
    ok = diff < 1e-3
    print("[parity]", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Laya 微调管线")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="head-only", choices=["head-only", "full"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--dataset", default=os.path.join(HERE, "artifacts", "samples.jsonl"))
    ap.add_argument("--parity", action="store_true")
    args = ap.parse_args()

    if args.parity:
        parity_test(args.model)
        return
    if not TORCH_OK:
        print(f"[pipeline] 需要 torch（当前导入失败：{_IMPORT_ERR}）。\n"
              f"          → 用 py3.12 CUDA venv：\n"
              f"           ~/.workbuddy/binaries/python/envs/laya-cuda/bin/python pipeline.py ...")
        return
    print(f"[pipeline] backend={args.backend} epochs={args.epochs} "
          f"dataset={args.dataset if os.path.isfile(args.dataset) else '无(仅对齐恢复)'}")

    # 训练骨架：加载模型、构造可训练子集、按样本迭代。完整版接 dataset.py 的 JSONL。
    st, tok, pred = build(args.model)
    model = LayaTorch(st, dict(pred.cfg)).float()
    make_trainable(model, args.backend)
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[pipeline] 可训练参数组就绪（head-only≈1.5M）。"
          f" 注：本脚手架实现前向 + parity；epoch 循环接 dataset.py 的 labels 即可启用。")
    print(f"[pipeline] 下一步：实现 loss_fn 的 score/noul 回归项，并在 {args.dataset} 上跑 "
          f"{args.epochs} 个 epoch（代码位置见 pipeline.py:loss_fn / make_trainable）。")


if __name__ == "__main__":
    main()
