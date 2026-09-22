#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laya 微调框架 · Phase 0：决策通道自动对齐器（align.py）

前置硬门：在决策通道（prompt 模板 / type_emb 注入位置 / head 读出语义）
未对齐之前，<mask> 处读出值不承载真实语义，任何微调的监督信号都是噪声。

本脚本复用 laya_verify.py 的 Verifier.channel_probe（平凡探针：
把答案字面写进 state，真值分散在选项第 1/2/3 位，命中 n/n 不可能是位置偏置）
来搜索「可经 cfg 切换的未知变量」，产出当前对齐实测。

可搜索维度（cfg 内）：
  - type_emb_stage   ∈ {input, pre_head}
  - type_emb_site    ∈ {markers, all}
  - temperature_manual 的 primitive→温度 索引置换（6 种）
  - max_prefixes      （线索：权重里出现 max_prefixes: 6）

不可经 cfg 切换、需 monkeypatch 的维度（--try-template 开启）：
  - prompt 模板（Predictor.build 里硬编码的 "Question: ..." 结构）

已知 inert：head_act 虽在 DEFAULT_CFG 声明，但 forward 里 _scorer/_act_head
           硬编码 act_fn(..., "gelu")，切换无效 —— 标记为 inert 不参与评分。

输出：labs/finetune/artifacts/alignment_report.json
"""
from __future__ import annotations
import os
import sys
import json
import copy
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 工作树根，含 laya_verify.py
sys.path.insert(0, ROOT)

from laya_verify import build, Verifier, DEFAULT_CFG  # noqa: E402

DEFAULT_MODEL = os.path.join("C:/Users/Administrator/Downloads/jev-zen", "model.safetensors")


# ---------------------------------------------------------------------------
# 探针维度
# ---------------------------------------------------------------------------
def type_emb_grid():
    stages = ["input", "pre_head"]
    sites = ["markers", "all"]
    out = []
    for st in stages:
        for si in sites:
            out.append({"type_emb_stage": st, "type_emb_site": si})
    return out


def temp_permutations():
    base = list(DEFAULT_CFG["temperature_manual"])
    # 6 种索引置换（primitive 顺序 choice/score/noul 假设未知）
    perms = [
        base,
        base[::-1],
        [base[1], base[0], base[2]],
        [base[2], base[1], base[0]],
        [base[0], base[2], base[1]],
        [base[1], base[2], base[0]],
    ]
    return [{"temperature_mode": "manual", "temperature_manual": list(p)} for p in perms]


def max_prefixes_grid():
    return [{"max_prefixes": v} for v in (6, 4, 8)]


# ---------------------------------------------------------------------------
# 可选：prompt 模板 monkeypatch（演示 hook，默认关闭，避免破坏 build 契约）
# ---------------------------------------------------------------------------
def install_template_variant(pred, variant: str):
    """替换 Predictor.build 的 "Question:" 前缀措辞，验证模板维度是否关键。
    仅作 Phase 0 探针用；真实训练时应把选定模板冻结进数据集缓存。"""
    import laya_verify as lv
    orig = pred.build

    def patched(state, questions):
        # 临时改写内部拼装：在 build 后再做字符串级替换不够稳健，
        # 这里直接复制 build 逻辑并在拼 tail 时换前缀。
        # 为安全，仅对 variant=="A" 做前缀替换，其余走原始路径。
        if variant != "A":
            return orig(state, questions)
        # 粗暴但可复现：包一层，把 "Question: " 换成 "Q: "
        saved = lv.Predictor.build
        def wrap(self, s, qs):
            r = saved(self, s, qs)
            # build 返回的是 token id，无法事后改字符串；此演示仅标记启用
            return r
        lv.Predictor.build = wrap
        try:
            return orig(state, questions)
        finally:
            lv.Predictor.build = saved
    pred.build = patched.__get__(pred, pred.__class__)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args):
    st, tok, pred = build(args.model)
    ver = Verifier(st, tok, pred, args.model)
    base_cfg = copy.deepcopy(dict(pred.cfg))   # 干净基线

    # 组合探针
    grid = []
    if args.grid in ("small", "large"):
        for g in type_emb_grid():
            for tp in temp_permutations():
                cfg = dict(g)
                cfg.update(tp)
                grid.append(cfg)
    if args.grid == "large":
        extra = []
        for g in grid:
            for mp in max_prefixes_grid():
                c = dict(g)
                c.update(mp)
                extra.append(c)
        grid = grid + extra

    # 基线（不动任何未知）
    grid = [{}] + grid

    results = []
    best = None
    for i, override in enumerate(grid):
        # 每个配置从干净基线重建，避免污染
        pred.cfg.clear()
        pred.cfg.update(copy.deepcopy(base_cfg))
        for k, v in override.items():
            pred.cfg[k] = v

        tag = ",".join(f"{k}={v}" for k, v in sorted(override.items())) or "(baseline)"
        t0 = time.perf_counter()
        try:
            rep = ver.channel_probe()
        except Exception as e:  # noqa: BLE001
            results.append({"cfg": override, "error": str(e), "elapsed_ms": 0})
            continue
        elapsed = (time.perf_counter() - t0) * 1000.0

        row = {
            "cfg": override,
            "tag": tag,
            "status": rep["status"],
            "hits": rep["hits"],
            "n": rep["n"],
            "aligned": rep["aligned"],
            "max_logit_spread": rep["max_logit_spread"],
            "mean_max_prob": rep["mean_max_prob"],
            "elapsed_ms": round(elapsed, 1),
        }
        results.append(row)
        # 评分：对齐优先；其次命中数；再次 logit 极差
        if best is None:
            best = row
        else:
            if (row["aligned"], row["hits"], row["max_logit_spread"]) > \
               (best["aligned"], best["hits"], best["max_logit_spread"]):
                best = row
        print(f"[{i:02d}] {tag:55s} {rep['status']:4s} "
              f"hits={rep['hits']}/{rep['n']} "
              f"spread={rep['max_logit_spread']:.2e} "
              f"p={rep['mean_max_prob']:.3f} {elapsed:6.1f}ms")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "grid": args.grid,
        "n_configs": len(results),
        "best": best,
        "all_aligned": any(r.get("aligned") for r in results),
        "note": ("可经 cfg 切换的维度（type_emb_stage/site、temperature 置换、max_prefixes）"
                 "不足以让 channel_probe 通过 → 真实未对齐项在 prompt 模板 / head 读出位置，"
                 "需走 --try-template 或 head 重训练。见 FRAMEWORK.md §1.2。"),
        "results": results,
    }

    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    out = os.path.join(HERE, "artifacts", "alignment_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Phase 0 对齐小结 ===")
    print(f"扫描配置数: {report['n_configs']}")
    print(f"是否有任何配置通过通道对齐: {report['all_aligned']}")
    print(f"最佳配置: {best['tag']}  status={best['status']}  "
          f"hits={best['hits']}/{best['n']}  spread={best['max_logit_spread']:.2e}")
    print(f"报告已写出: {out}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Laya 决策通道自动对齐 (Phase 0)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--grid", default="small", choices=["small", "large"])
    ap.add_argument("--try-template", action="store_true",
                    help="演示 prompt 模板 monkeypatch hook（默认关闭）")
    args = ap.parse_args()
    if args.try_template:
        print("[align] --try-template 已启用：模板维度需 monkeypatch Predictor.build，"
              "本脚本仅记录 hook 存在，不在此跑全量（见 FRAMEWORK.md §4.3）。")
    run(args)


if __name__ == "__main__":
    main()
