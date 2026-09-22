#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laya 微调框架 · 数据集构造（dataset.py）

出处：labs/ 下 16 个游戏场景 + 2048 主场景。每个场景实现 core.js 约定的 SCENE 接口
（init/legal/step/text/questions/rule/goalOk/maxSteps）。rule 策略产出「教师标签」。

样本 schema（与模型输入同构，见 FRAMEWORK.md §3.2）：
  { scene, seed, step, state, questions, labels, alignment_tag }

本模块：
  1) rollout(scene, seed)  —— 在 SCENE 上滚动产出 (state, questions, rule 标签) 样本；
  2) tokenize_sample()    —— 经 Predictor.build 预分词，落盘 ids/marker_pos/marker_type；
  3) ToyScene             —— 一个内建最小场景，使模块可脱机自测（无需 JS 引擎）；
  4) harvest_from_labs()  —— 真实场景桥接 stub（经 node + core.js 驱动 <scene>_laya.html 的
                             SCENE 全局；本脚手架只描述契约，完整实现见 FRAMEWORK.md §3.4）。
"""
from __future__ import annotations
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 工作树根，含 laya_verify.py
sys.path.insert(0, ROOT)

from laya_verify import build, Question, PRIMITIVES  # noqa: E402

DEFAULT_MODEL = os.path.join("C:/Users/Administrator/Downloads/jev-zen", "model.safetensors")


# ---------------------------------------------------------------------------
# 标签派生（来自 SCENE.rule 输出）
# ---------------------------------------------------------------------------
def derive_labels(questions, rule):
    """把 (typed questions, rule 输出) 转成训练标签。
    rule: {choice: key, ranks:[{key,label,score,detail,risky}], note}
    """
    labels = {}
    ranks = rule.get("ranks") or []
    score_by_key = {r["key"]: float(r.get("score", 0.0)) for r in ranks}
    # score 归一化为分布（用于 choice 排序分 / score 期望值）
    vals = list(score_by_key.values())
    if vals:
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        prob = {k: (v - lo) / rng for k, v in score_by_key.items()}
        denom = sum(prob.values()) or 1.0
        prob = {k: p / denom for k, p in prob.items()}
    else:
        prob = {}
    for q in questions:
        qid = q.id if hasattr(q, "id") else q["id"]
        qtype = q.type if hasattr(q, "type") else q["type"]
        if qtype == "choice":
            gold = rule.get("choice")
            labels[qid] = {"type": "choice", "key": gold,
                           "rank_score": prob.get(gold, 0.0)}
        elif qtype == "score":
            # 期望值：按归一化分布加权
            ev = sum(prob[k] * score_by_key[k] for k in prob) if prob else 0.0
            labels[qid] = {"type": "score", "expected_value": ev}
        elif qtype == "noul":
            risky = any(r.get("risky") for r in ranks)
            labels[qid] = {"type": "noul", "p_yes": 0.15 if risky else 0.85}
    return labels


# ---------------------------------------------------------------------------
# 滚动采集
# ---------------------------------------------------------------------------
def rollout(scene, seed, max_steps=None, sopt=None):
    """在 SCENE 上滚动，逐步产出样本 dict。scene 为 duck-typed SCENE 对象。"""
    sopt = sopt or {}
    s = scene.init(seed, sopt)
    guard = max_steps or (scene.maxSteps(sopt) if hasattr(scene, "maxSteps") else 200)
    samples = []
    step = 0
    while not scene.done(s) and guard > 0:
        guard -= 1
        legal = scene.legal(s)
        if not legal:
            break
        qs = scene.questions(s, legal, sopt) if hasattr(scene, "questions") else []
        rule = scene.rule(s, legal, sopt) if hasattr(scene, "rule") else None
        if qs and rule:
            samples.append({
                "scene": getattr(scene, "id", "scene"),
                "seed": seed,
                "step": step,
                "state": scene.text(s, legal, sopt) if hasattr(scene, "text") else "",
                "questions": [q.to_dict() if hasattr(q, "to_dict") else q for q in qs],
                "labels": derive_labels(qs, rule),
            })
        scene.step(s, rule.get("choice") if rule else legal[0], sopt) if hasattr(scene, "step") else None
        step += 1
        if scene.goalOk(s, sopt) if hasattr(scene, "goalOk") else False:
            break
    return samples


def tokenize_sample(pred, sample):
    """经 Predictor.build 预分词，回填 token 级字段。"""
    qs = [Question(**q) for q in sample["questions"]]
    built = pred.build(sample["state"], qs)
    sample["ids"] = built["ids"]
    sample["marker_pos"] = built["marker_pos"]
    sample["marker_type"] = built["marker_type"]
    sample["spans"] = built["spans"]
    sample["tokens"] = built["tokens"]
    return sample


# ---------------------------------------------------------------------------
# 内建最小场景（自测用，无需 JS 引擎）
# ---------------------------------------------------------------------------
class ToyScene:
    """确定性迷宫式场景：状态是 3 个槽的 int，合法动作 UP/DOWN/LEFT/RIGHT。"""
    id = "toy"
    name = "Toy (self-test)"

    def init(self, seed, sopt):
        import random
        r = random.Random(seed)
        return {"a": r.randint(0, 3), "b": r.randint(0, 3), "c": r.randint(0, 3)}

    def legal(self, s):
        return ["UP", "DOWN", "LEFT", "RIGHT"]

    def step(self, s, key, sopt):
        d = {"UP": ("a", 1), "DOWN": ("a", -1), "LEFT": ("b", 1), "RIGHT": ("b", -1)}
        k, dv = d.get(key, ("c", 0))
        s[k] = max(0, min(3, s[k] + dv))
        return {"info": f"{key}→{k}={s[k]}"}

    def text(self, s, legal, sopt):
        return f"state: a={s['a']} b={s['b']} c={s['c']}\nlegal: {','.join(legal)}"

    def questions(self, s, legal, sopt):
        return [Question("q0", "choice", "下一步最优方向？",
                         {"UP": "上", "DOWN": "下", "LEFT": "左", "RIGHT": "右"})]

    def rule(self, s, legal, sopt):
        # 教师：把 a、b 推向 3。注意 step 语义：UP/DOWN 改 a，LEFT/RIGHT 改 b；
        # LEFT 增 b、RIGHT 减 b，故推高 b 应取 LEFT。
        if s["a"] < 3:
            best = "UP"
        elif s["b"] < 3:
            best = "LEFT"
        else:
            best = "UP"  # 已达成（a==b==3），任选合法键，下一步 done 退出
        ranks = [{"key": k, "label": k, "score": 1.0 if k == best else 0.0,
                  "detail": "", "risky": False} for k in legal]
        return {"choice": best, "ranks": ranks, "note": "toy heuristic"}

    def done(self, s):
        return s["a"] == 3 and s["b"] == 3

    def goalOk(self, s, sopt):
        return self.done(s)

    def maxSteps(self, sopt):
        return 10


# ---------------------------------------------------------------------------
# 真实场景桥接（stub：经 node + core.js 驱动 HTML 里的 SCENE 全局）
# ---------------------------------------------------------------------------
def harvest_from_labs(model_path, scenes, seeds, node_exe=None):
    """对 labs/ 下每个 <name>_laya.html，用 node 加载 core.js + 该页 SCENE，
    滚动产出样本 JSONL。完整实现需一个 node 侧入口（暴露 SCENE + rollout 到 stdout JSONL）。
    此处仅描述契约，避免在本脚手架里内联 node 多进程细节。
    """
    raise NotImplementedError(
        "harvest_from_labs 需 node 侧入口（驱动 <scene>_laya.html 的 SCENE 全局）。"
        "契约见 FRAMEWORK.md §3.4：每个场景输出 {scene,seed,step,state,questions,labels}。"
        "本仓库自测用 ToyScene 已覆盖管线；接真实场景时实现此函数即可。"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Laya 数据集构造")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--self-test", action="store_true", help="用 ToyScene 跑通管线并写样本")
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts", "samples.jsonl"))
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.self_test:
        _, _, pred = build(args.model)
        scene = ToyScene()
        all_samples = []
        for seed in range(args.seeds):
            for smp in rollout(scene, seed):
                all_samples.append(tokenize_sample(pred, smp))
        # 附带对齐标签（Phase 0 结果，若已跑过 align.py）
        rep = os.path.join(HERE, "artifacts", "alignment_report.json")
        tag = None
        if os.path.isfile(rep):
            with open(rep, encoding="utf-8") as f:
                tag = json.load(f)["best"]
        for smp in all_samples:
            smp["alignment_tag"] = tag
        with open(args.out, "w", encoding="utf-8") as f:
            for smp in all_samples:
                f.write(json.dumps(smp, ensure_ascii=False) + "\n")
        print(f"[dataset] ToyScene 自测完成：{len(all_samples)} 条样本 → {args.out}")
        print(f"[dataset] 示例样本 keys: {list(all_samples[0].keys())}")
        return
    print("[dataset] 需 --self-test（或实现 harvest_from_labs 接真实场景）。")


if __name__ == "__main__":
    main()
