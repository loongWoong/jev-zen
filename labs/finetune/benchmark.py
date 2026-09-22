#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laya 微调框架 · 评测台（benchmark.py）

对齐门（前置，必须 PASS 否则下游指标不可信）：Verifier.channel_probe()
指标：
  - 对齐状态 / logit 极差（通道增益）
  - 延迟：Verifier.benchmark（1/5/10 问题，ms/问题）
  - 标签准确率：测试集 top-1 vs rule 教师标签（微调目标代理）
  - 任务迁移 / 泛化：见 FRAMEWORK.md §5（需接 labs 场景 rollout，本脚手架提供 label-acc 代理）

输出：labs/finetune/artifacts/benchmark_<tag>.json
"""
from __future__ import annotations
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 工作树根，含 laya_verify.py
sys.path.insert(0, ROOT)

from laya_verify import build, Verifier, Question  # noqa: E402

DEFAULT_MODEL = os.path.join("C:/Users/Administrator/Downloads/jev-zen", "model.safetensors")


def eval_label_accuracy(pred, jsonl_path, limit=200):
    """测试集 top-1 选项 vs rule 教师标签。读 tokenized 样本（ids/marker_pos/marker_type）。"""
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return None
    n = 0
    hit_choice = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if limit and n >= limit:
                break
            smp = json.loads(line)
            ids = smp.get("ids")
            if ids is None:
                continue
            qs = [Question(**q) for q in smp["questions"]]
            out = pred.model.forward(ids, smp["marker_pos"], smp["marker_type"])
            logits = out["logits"]
            cursor = 0
            for q in qs:
                n_opt = len(q.options())
                lg = logits[cursor:cursor + n_opt]
                cursor += n_opt
                gold = (smp.get("labels") or {}).get(q.id, {}).get("key")
                if gold is None:
                    continue
                pred_key = q.options()[int(__import__("numpy").argmax(lg))][0]
                n += 1
                if pred_key == gold:
                    hit_choice += 1
    return {"n": n, "choice_top1_acc": (hit_choice / n) if n else None}


def run(model_path, dataset=None, tag="before"):
    st, tok, pred = build(model_path)
    ver = Verifier(st, tok, pred, model_path)

    report = {"tag": tag, "model": model_path, "metrics": {}}

    # 1) 对齐门
    ch = ver.channel_probe()
    report["metrics"]["channel"] = {
        "status": ch["status"], "aligned": ch["aligned"],
        "hits": ch["hits"], "n": ch["n"],
        "max_logit_spread": ch["max_logit_spread"],
        "mean_max_prob": ch["mean_max_prob"],
    }

    # 2) 延迟
    try:
        bm = ver.benchmark()
        report["metrics"]["latency"] = bm
    except Exception as e:  # noqa: BLE001
        report["metrics"]["latency"] = {"error": str(e)}

    # 3) 标签准确率（若有测试集）
    if dataset:
        la = eval_label_accuracy(pred, dataset)
        if la:
            report["metrics"]["label_accuracy"] = la

    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    out = os.path.join(HERE, "artifacts", f"benchmark_{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"=== benchmark[{tag}] ===")
    print(f"对齐: {ch['status']} (aligned={ch['aligned']}) hits={ch['hits']}/{ch['n']} "
          f"spread={ch['max_logit_spread']:.2e}")
    if dataset and report["metrics"].get("label_accuracy"):
        acc = report["metrics"]["label_accuracy"]["choice_top1_acc"]
        print(f"标签准确率(choice top1): {acc:.3f}  (n={report['metrics']['label_accuracy']['n']})")
    print(f"报告: {out}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Laya 评测台")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", default=os.path.join(HERE, "artifacts", "samples.jsonl"))
    ap.add_argument("--tag", default="before")
    args = ap.parse_args()
    run(args.model, args.dataset if os.path.isfile(args.dataset) else None, args.tag)


if __name__ == "__main__":
    main()
